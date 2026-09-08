#!/usr/bin/env python3
"""Post-hoc global refinement with the native GraphDECO maintenance schedule.

The online mapping stage is unchanged. At the online endpoint this runner:

1. evaluates all Train/Test views with PSNR/SSIM/LPIPS(VGG);
2. performs one extra GaussianModel.reset_opacity() requested by our protocol;
3. continues the existing online Adam optimizer state while restarting the
   refinement-stage xyz learning-rate index from iteration 1;
4. follows the original GraphDECO densification/pruning/reset schedule during
   post-hoc refinement:
      - collect densification statistics while iteration < 15000;
      - densify/prune when iteration > 500 and iteration % 100 == 0;
      - prune opacity < 0.005;
      - use max_screen_size=20 only when iteration > 3000;
      - reset opacity every 3000 iterations, while inside the densification
        window (3000/6000/9000/12000 for the default 15000 stop);
5. evaluates Train/Test at exact optimizer-update checkpoints.

The structural schedule is relative to the post-hoc refinement stage, not the
preceding online global-iteration counter. Camera poses remain fixed.
"""
from __future__ import annotations

import json
import os
import random
import time
from typing import Any

import torch

import run_pipeline_execution_benchmark_repro_empty_safe as safe
import run_pipeline_execution_benchmark_repro_posthoc_global_refine_checkpoints_lpips as quality_helpers


def _parse_positive_int_list(name: str, default: str) -> tuple[int, ...]:
    raw = os.environ.get(name, default)
    values: list[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value <= 0:
            raise ValueError(f"{name} values must be positive, got {value}")
        values.append(value)
    if not values:
        raise ValueError(f"{name} is empty")
    return tuple(sorted(set(values)))


def _install_native_graphdeco_refinement() -> None:
    from async_pipeline.backend_evaluation import BackendEvaluationMixin

    original_finalize_impl = BackendEvaluationMixin._finalize_impl

    checkpoints = _parse_positive_int_list(
        "PIPELINE_GLOBAL_REFINE_ITER_CHECKPOINTS",
        "10000,20000,30000,40000,50000,60000,70000,80000,90000,100000",
    )
    total_iterations = int(
        os.environ.get("PIPELINE_GLOBAL_REFINE_TOTAL_ITERS", str(max(checkpoints)))
    )
    if total_iterations <= 0:
        raise ValueError("PIPELINE_GLOBAL_REFINE_TOTAL_ITERS must be positive")
    if max(checkpoints) > total_iterations:
        raise ValueError(
            "largest PIPELINE_GLOBAL_REFINE_ITER_CHECKPOINTS value exceeds "
            "PIPELINE_GLOBAL_REFINE_TOTAL_ITERS"
        )

    densify_from = int(os.environ.get("PIPELINE_NATIVE_DENSIFY_FROM_ITER", "500"))
    densify_until = int(os.environ.get("PIPELINE_NATIVE_DENSIFY_UNTIL_ITER", "15000"))
    densification_interval = int(
        os.environ.get("PIPELINE_NATIVE_DENSIFICATION_INTERVAL", "100")
    )
    opacity_reset_interval = int(
        os.environ.get("PIPELINE_NATIVE_OPACITY_RESET_INTERVAL", "3000")
    )
    min_opacity = float(os.environ.get("PIPELINE_NATIVE_MIN_OPACITY", "0.005"))
    max_screen_after_reset = float(
        os.environ.get("PIPELINE_NATIVE_MAX_SCREEN_SIZE", "20")
    )
    seed = int(
        os.environ.get(
            "PIPELINE_GLOBAL_REFINE_SEED",
            os.environ.get("PIPELINE_BENCHMARK_SEED", "0"),
        )
    )

    if densify_from < 0 or densify_until <= densify_from:
        raise ValueError("invalid native densification interval")
    if densification_interval <= 0 or opacity_reset_interval <= 0:
        raise ValueError("native maintenance intervals must be positive")
    if not 0.0 < min_opacity < 1.0:
        raise ValueError("PIPELINE_NATIVE_MIN_OPACITY must be in (0,1)")

    def _metric_line(label: str, train: dict[str, Any], test: dict[str, Any]) -> None:
        def fmt(metrics: dict[str, Any]) -> str:
            if not metrics or int(metrics.get("num_views", 0)) <= 0:
                return "missing"
            return (
                f"{float(metrics['psnr']):.3f}/"
                f"{float(metrics['ssim']):.4f}/"
                f"{float(metrics['lpips']):.4f}"
            )

        print(
            f"[native3dgs/{label}] Train={fmt(train)} Test={fmt(test)}",
            flush=True,
        )

    def finalize_with_native_graphdeco(self):
        if (
            self.train_packet_count <= 0
            or self.gaussians is None
            or int(self.gaussians.get_xyz.shape[0]) == 0
        ):
            return original_finalize_impl(self)

        from utils.loss_utils import l1_loss, ssim

        ntrain = len(self.train_cameras)
        ntest = len(self.test_cameras)
        if ntrain <= 0:
            return original_finalize_impl(self)

        g_online = int(self.gaussians.get_xyz.shape[0])
        online_iteration_end = int(self.global_iteration)
        scene_extent = float(self._current_scene_extent())
        output_name = "posthoc_global_refinement_native3dgs_metrics.json"
        event_name = "posthoc_global_refinement_native3dgs_events.json"

        print(
            "\n[native3dgs] post-hoc protocol: "
            f"train_views={ntrain} test_views={ntest} total_iters={total_iterations} "
            f"checkpoints={list(checkpoints)} G_online={g_online}",
            flush=True,
        )
        print(
            "[native3dgs] schedule: extra reset@0; "
            f"densify if iter>{densify_from}, every {densification_interval}, "
            f"until iter<{densify_until}; opacity<{min_opacity}; "
            f"reset every {opacity_reset_interval}; max_screen={max_screen_after_reset} "
            f"only after iter>{opacity_reset_interval}",
            flush=True,
        )

        online_train = self._evaluate(self.train_cameras)
        online_test = self._evaluate(self.test_cameras)
        _metric_line("online", online_train, online_test)

        events: list[dict[str, Any]] = []
        checkpoints_payload: dict[str, Any] = {}

        def save_events() -> None:
            self._save_json(event_name, events)

        def save_progress(completed: int) -> None:
            payload = {
                "protocol": "online endpoint + native GraphDECO post-hoc refinement schedule",
                "checkpoint_iterations": list(checkpoints),
                "total_refinement_iterations_requested": total_iterations,
                "completed_refinement_iterations": completed,
                "num_train_views": ntrain,
                "num_test_views": ntest,
                "continue_online_optimizer": True,
                "restart_refinement_xyz_lr_index": True,
                "poses_fixed": True,
                "scene_extent": scene_extent,
                "native_schedule": {
                    "extra_initial_opacity_reset": True,
                    "densify_from_iter": densify_from,
                    "densify_until_iter": densify_until,
                    "densification_interval": densification_interval,
                    "densify_grad_threshold": float(self.opt.densify_grad_threshold),
                    "min_opacity": min_opacity,
                    "opacity_reset_interval": opacity_reset_interval,
                    "max_screen_size_after_first_reset": max_screen_after_reset,
                },
                "gaussians_online": g_online,
                "gaussians_current": int(self.gaussians.get_xyz.shape[0]),
                "online": {"train": online_train, "test": online_test},
                "checkpoints": checkpoints_payload,
            }
            self._save_json(output_name, payload)

        # Protocol-specific extra reset before the otherwise-native schedule.
        opacity_before = self.gaussians.get_opacity.detach()
        initial_reset_event = {
            "type": "opacity_reset_initial",
            "refinement_iteration": 0,
            "count_before": int(self.gaussians.get_xyz.shape[0]),
            "opacity_before_mean": float(opacity_before.mean().item()),
            "opacity_before_max": float(opacity_before.max().item()),
        }
        self.gaussians.reset_opacity()
        opacity_after = self.gaussians.get_opacity.detach()
        initial_reset_event.update(
            {
                "count_after": int(self.gaussians.get_xyz.shape[0]),
                "opacity_after_mean": float(opacity_after.mean().item()),
                "opacity_after_max": float(opacity_after.max().item()),
            }
        )
        events.append(initial_reset_event)
        self.gaussians.optimizer.zero_grad(set_to_none=True)
        save_events()
        save_progress(0)

        rng = random.Random(seed)
        camera_stack = list(range(ntrain))
        refinement_start = time.perf_counter()

        for refinement_iteration in range(1, total_iterations + 1):
            self.global_iteration += 1
            self.gaussians.update_learning_rate(refinement_iteration)

            if not camera_stack:
                camera_stack = list(range(ntrain))
            stack_pos = rng.randrange(len(camera_stack))
            camera_index = camera_stack.pop(stack_pos)
            camera = self.train_cameras[camera_index]

            background = (
                torch.rand(3, device=self.device)
                if self.opt.random_background
                else self.background
            )
            render_pkg = self.render(
                camera,
                self.gaussians,
                self.pipe,
                background,
                use_trained_exp=False,
                separate_sh=False,
            )
            image = render_pkg["render"]
            gt = camera.original_image
            if camera.alpha_mask is not None:
                image = image * camera.alpha_mask

            ll1 = l1_loss(image, gt)
            ssim_value = ssim(image, gt)
            loss = (
                (1.0 - self.opt.lambda_dssim) * ll1
                + self.opt.lambda_dssim * (1.0 - ssim_value)
            )
            loss.backward()

            with torch.no_grad():
                # Match the original GraphDECO ordering: densification statistics,
                # densify/prune, opacity reset, then optimizer.step().
                if refinement_iteration < densify_until:
                    count_before_stats = int(self.gaussians.get_xyz.shape[0])
                    indices = self._visibility_indices(
                        render_pkg["visibility_filter"], count_before_stats
                    )
                    radii = render_pkg["radii"]
                    if indices.numel() > 0:
                        self.gaussians.max_radii2D[indices] = torch.maximum(
                            self.gaussians.max_radii2D[indices], radii[indices]
                        )
                        self.gaussians.add_densification_stats(
                            render_pkg["viewspace_points"], indices
                        )

                    if (
                        refinement_iteration > densify_from
                        and refinement_iteration % densification_interval == 0
                    ):
                        count_before = int(self.gaussians.get_xyz.shape[0])
                        size_threshold = (
                            max_screen_after_reset
                            if refinement_iteration > opacity_reset_interval
                            else None
                        )
                        maintenance_start = time.perf_counter()
                        self.gaussians.densify_and_prune(
                            self.opt.densify_grad_threshold,
                            min_opacity,
                            scene_extent,
                            size_threshold,
                            radii,
                        )
                        if torch.cuda.is_available():
                            torch.cuda.current_stream(self.device).synchronize()
                        maintenance_sec = time.perf_counter() - maintenance_start
                        count_after = int(self.gaussians.get_xyz.shape[0])
                        event = {
                            "type": "densify_and_prune",
                            "refinement_iteration": refinement_iteration,
                            "count_before": count_before,
                            "count_after": count_after,
                            "delta": count_after - count_before,
                            "grad_threshold": float(self.opt.densify_grad_threshold),
                            "min_opacity": min_opacity,
                            "scene_extent": scene_extent,
                            "max_screen_size": size_threshold,
                            "maintenance_sec": maintenance_sec,
                        }
                        events.append(event)
                        print(
                            "[native3dgs/maintenance] "
                            f"iter={refinement_iteration} G={count_before}->{count_after} "
                            f"max_screen={size_threshold}",
                            flush=True,
                        )

                    if refinement_iteration % opacity_reset_interval == 0:
                        reset_before = self.gaussians.get_opacity.detach()
                        event = {
                            "type": "opacity_reset",
                            "refinement_iteration": refinement_iteration,
                            "count_before": int(self.gaussians.get_xyz.shape[0]),
                            "opacity_before_mean": float(reset_before.mean().item()),
                            "opacity_before_max": float(reset_before.max().item()),
                        }
                        self.gaussians.reset_opacity()
                        reset_after = self.gaussians.get_opacity.detach()
                        event.update(
                            {
                                "count_after": int(self.gaussians.get_xyz.shape[0]),
                                "opacity_after_mean": float(reset_after.mean().item()),
                                "opacity_after_max": float(reset_after.max().item()),
                            }
                        )
                        events.append(event)
                        print(
                            f"[native3dgs/reset] iter={refinement_iteration}",
                            flush=True,
                        )

                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)

            if (
                refinement_iteration % 1000 == 0
                or refinement_iteration in checkpoints
                or refinement_iteration == total_iterations
            ):
                save_events()
                save_progress(refinement_iteration)

            if refinement_iteration in checkpoints:
                train_metrics = self._evaluate(self.train_cameras)
                test_metrics = self._evaluate(self.test_cameras)
                checkpoints_payload[str(refinement_iteration)] = {
                    "refinement_iteration": refinement_iteration,
                    "train": train_metrics,
                    "test": test_metrics,
                    "num_gaussians": int(self.gaussians.get_xyz.shape[0]),
                    "elapsed_update_and_metric_wall_sec": time.perf_counter()
                    - refinement_start,
                }
                _metric_line(
                    f"iter{refinement_iteration}", train_metrics, test_metrics
                )
                save_progress(refinement_iteration)

        final_key = str(max(checkpoints))
        if final_key not in checkpoints_payload:
            # This can only happen when the requested total iterations exceed the
            # last metric checkpoint. Preserve the last available metric endpoint.
            final_key = str(checkpoints[-1])
        final_metrics = checkpoints_payload[final_key]

        endpoint_metrics = {
            "protocol": "online endpoint + native GraphDECO post-hoc refinement schedule",
            "checkpoint_iterations": list(checkpoints),
            "total_refinement_iterations": total_iterations,
            "num_train_views": ntrain,
            "num_test_views": ntest,
            "continue_online_optimizer": True,
            "restart_refinement_xyz_lr_index": True,
            "poses_fixed": True,
            "gaussians_online": g_online,
            "gaussians_final": int(self.gaussians.get_xyz.shape[0]),
            "scene_extent": scene_extent,
            "online": {"train": online_train, "test": online_test},
            "checkpoints": checkpoints_payload,
            "native_schedule": {
                "extra_initial_opacity_reset": True,
                "densify_from_iter": densify_from,
                "densify_until_iter": densify_until,
                "densification_interval": densification_interval,
                "densify_grad_threshold": float(self.opt.densify_grad_threshold),
                "min_opacity": min_opacity,
                "opacity_reset_interval": opacity_reset_interval,
                "max_screen_size_after_first_reset": max_screen_after_reset,
            },
        }
        self._save_json(output_name, endpoint_metrics)
        save_events()

        summary = {
            "backend": "StreamingIncrementalBackend",
            "coordinate_contract": (
                "ReSplat packet is left-camera local; backend applies metric OpenCV Twc "
                "to means and rotations before insertion"
            ),
            "num_train_packets": self.train_packet_count,
            "num_train_cameras": ntrain,
            "num_test_cameras": ntest,
            "total_iterations": self.global_iteration,
            "online_iteration_end": online_iteration_end,
            "final_num_gaussians": int(self.gaussians.get_xyz.shape[0]),
            "spatial_lr_scale": self.config.spatial_lr_scale,
            "num_skipped_invalid_poses": len(self.skipped_pose_log),
            "evaluation_enabled": self.config.evaluation_enabled,
            "write_runtime_artifacts": self.config.write_runtime_artifacts,
            "final_metrics": {
                "train_inserted": final_metrics["train"],
                "test_all": final_metrics["test"],
            },
            "online_metrics": {
                "train_inserted": online_train,
                "test_all": online_test,
            },
            "posthoc_global_refinement": endpoint_metrics,
            "gpu_memory": self._gpu_memory_stats(),
            "wall_time_sec": time.perf_counter() - self.wall_start,
        }
        if self.config.save_final_ply:
            self._save_point_cloud(self.global_iteration)
        self._save_json("incremental_backend_summary.json", summary)
        if self.wandb_run is not None:
            self.wandb_run.finish()
        return summary

    BackendEvaluationMixin._finalize_impl = finalize_with_native_graphdeco


if __name__ == "__main__":
    quality_helpers._disable_intermediate_online_evaluation()
    quality_helpers._install_lpips_evaluator()
    _install_native_graphdeco_refinement()
    safe.repro.main()

#!/usr/bin/env python3
"""Quality runner for continuous post-hoc refinement with pass + update checkpoints.

The online mapping stage is unchanged. At finalize this runner:
1. evaluates the Online Train/Test endpoint with PSNR/SSIM/LPIPS(VGG);
2. performs one native GaussianModel.reset_opacity();
3. continues the existing online Adam state;
4. runs ONE continuous shuffled-view refinement trajectory;
5. evaluates at requested full-pass checkpoints and/or exact optimizer-update
   checkpoints.

Environment variables:
  PIPELINE_GLOBAL_REFINE_CHECKPOINT_PASSES
      Comma-separated full-pass checkpoints, e.g. 50,60,...,150. Empty disables.
  PIPELINE_GLOBAL_REFINE_CHECKPOINT_ITERS
      Comma-separated exact refinement-update checkpoints, e.g.
      20000,30000,40000,50000. Empty disables.
  PIPELINE_GLOBAL_REFINE_SEED
      Shuffle seed; falls back to PIPELINE_BENCHMARK_SEED.

Gaussian topology and poses are fixed during post-hoc refinement. There is exactly
one opacity reset at the beginning of the continuous trajectory.
"""
from __future__ import annotations

import os
import random
import time
from typing import Any

import torch

import run_pipeline_execution_benchmark_repro_empty_safe as safe


def _disable_intermediate_online_evaluation() -> None:
    from async_pipeline.backend_evaluation import BackendEvaluationMixin

    def no_intermediate_evaluation(self, stage, update, active_cameras):
        return None

    BackendEvaluationMixin._record_evaluation = no_intermediate_evaluation


def _install_lpips_evaluator() -> None:
    from async_pipeline.backend_evaluation import BackendEvaluationMixin

    try:
        from lpips import LPIPS
    except ImportError as exc:
        raise RuntimeError(
            "LPIPS evaluation requires package 'lpips'. Install with: pip install lpips"
        ) from exc

    @torch.no_grad()
    def evaluate_with_lpips(self, cameras):
        from utils.image_utils import psnr
        from utils.loss_utils import l1_loss, ssim

        selected = list(cameras)
        if self.config.eval_max_views > 0:
            selected = selected[: self.config.eval_max_views]
        if not selected:
            return {"num_views": 0}

        model = getattr(self, "_final_lpips_vgg", None)
        if model is None:
            model = LPIPS(net="vgg").to(self.device).eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            setattr(self, "_final_lpips_vgg", model)

        l1_sum = psnr_sum = ssim_sum = lpips_sum = 0.0
        for camera in selected:
            image = self.render(
                camera,
                self.gaussians,
                self.pipe,
                self.background,
                use_trained_exp=False,
                separate_sh=False,
            )["render"].clamp(0.0, 1.0)
            gt = camera.original_image.clamp(0.0, 1.0)
            l1_sum += float(l1_loss(image, gt).mean().item())
            psnr_sum += float(psnr(image, gt).mean().item())
            ssim_sum += float(ssim(image, gt).mean().item())
            lpips_sum += float(
                model(gt.unsqueeze(0), image.unsqueeze(0), normalize=True).mean().item()
            )

        count = len(selected)
        return {
            "num_views": count,
            "l1": l1_sum / count,
            "psnr": psnr_sum / count,
            "ssim": ssim_sum / count,
            "lpips": lpips_sum / count,
        }

    BackendEvaluationMixin._evaluate = evaluate_with_lpips


def _parse_positive_ints(name: str, default: str = "") -> tuple[int, ...]:
    raw = os.environ.get(name, default).strip()
    if not raw:
        return ()
    values: list[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value <= 0:
            raise ValueError(f"{name} values must be positive, got {value}")
        values.append(value)
    return tuple(sorted(set(values)))


def _install_dual_checkpoint_refinement() -> None:
    from async_pipeline.backend_evaluation import BackendEvaluationMixin

    original_finalize_impl = BackendEvaluationMixin._finalize_impl
    pass_checkpoints = _parse_positive_ints(
        "PIPELINE_GLOBAL_REFINE_CHECKPOINT_PASSES",
        "50,60,70,80,90,100,110,120,130,140,150",
    )
    iter_checkpoints = _parse_positive_ints(
        "PIPELINE_GLOBAL_REFINE_CHECKPOINT_ITERS",
        "20000,30000,40000,50000",
    )
    if not pass_checkpoints and not iter_checkpoints:
        raise ValueError("At least one global-refinement checkpoint must be requested")

    seed = int(
        os.environ.get(
            "PIPELINE_GLOBAL_REFINE_SEED",
            os.environ.get("PIPELINE_BENCHMARK_SEED", "0"),
        )
    )

    def _fmt(metrics: dict[str, Any]) -> str:
        if not metrics or int(metrics.get("num_views", 0)) <= 0:
            return "missing"
        return (
            f"{float(metrics['psnr']):.3f}/"
            f"{float(metrics['ssim']):.4f}/"
            f"{float(metrics['lpips']):.4f}"
        )

    def _print_metrics(label: str, train: dict[str, Any], test: dict[str, Any]) -> None:
        print(
            f"[final-quality/{label}] Train={_fmt(train)} Test={_fmt(test)}",
            flush=True,
        )

    def finalize_with_dual_checkpoints(self):
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

        max_from_pass = (max(pass_checkpoints) * ntrain) if pass_checkpoints else 0
        max_from_iter = max(iter_checkpoints) if iter_checkpoints else 0
        stop_iterations = max(max_from_pass, max_from_iter)
        max_pass_needed = (stop_iterations + ntrain - 1) // ntrain
        g_before = int(self.gaussians.get_xyz.shape[0])
        online_iteration_end = int(self.global_iteration)
        output_name = "posthoc_global_refinement_dual_checkpoint_metrics.json"

        print(
            "\n[final-quality] dual-checkpoint protocol: "
            f"train_views={ntrain} test_views={ntest} "
            f"pass_checkpoints={list(pass_checkpoints)} "
            f"iter_checkpoints={list(iter_checkpoints)} "
            f"stop_iterations={stop_iterations} G={g_before}",
            flush=True,
        )

        online_train = self._evaluate(self.train_cameras)
        online_test = self._evaluate(self.test_cameras)
        _print_metrics("online", online_train, online_test)

        if not hasattr(self.gaussians, "reset_opacity"):
            raise RuntimeError("GaussianModel has no reset_opacity()")
        opacity_before = self.gaussians.get_opacity.detach().clone()
        self.gaussians.reset_opacity()
        opacity_after = self.gaussians.get_opacity.detach()
        reset_stats = {
            "enabled": True,
            "implementation": "GaussianModel.reset_opacity()",
            "before_mean": float(opacity_before.mean().item()),
            "before_max": float(opacity_before.max().item()),
            "after_mean": float(opacity_after.mean().item()),
            "after_max": float(opacity_after.max().item()),
        }

        self.gaussians.optimizer.zero_grad(set_to_none=True)
        rng = random.Random(seed)
        refinement_iteration = 0
        by_pass: dict[str, Any] = {}
        by_iteration: dict[str, Any] = {}
        metrics_cache: dict[int, dict[str, Any]] = {}

        def evaluate_current(label: str) -> dict[str, Any]:
            cached = metrics_cache.get(refinement_iteration)
            if cached is None:
                train_metrics = self._evaluate(self.train_cameras)
                test_metrics = self._evaluate(self.test_cameras)
                cached = {
                    "refinement_iterations": refinement_iteration,
                    "equivalent_passes": refinement_iteration / ntrain,
                    "train": train_metrics,
                    "test": test_metrics,
                }
                metrics_cache[refinement_iteration] = cached
                _print_metrics(label, train_metrics, test_metrics)
            else:
                print(
                    f"[final-quality/{label}] reused metrics at iter={refinement_iteration}",
                    flush=True,
                )
            return dict(cached)

        def save_progress() -> None:
            payload = {
                "protocol": "online endpoint + one continuous post-hoc trajectory with pass and exact-update checkpoints",
                "checkpoint_passes": list(pass_checkpoints),
                "checkpoint_iterations": list(iter_checkpoints),
                "num_train_views": ntrain,
                "num_test_views": ntest,
                "stop_refinement_iterations": stop_iterations,
                "continue_online_optimizer": True,
                "restart_refinement_xyz_lr_index": True,
                "opacity_reset": reset_stats,
                "topology_fixed": True,
                "densification": False,
                "pruning": False,
                "gaussians": g_before,
                "online": {"train": online_train, "test": online_test},
                "checkpoints_by_pass": by_pass,
                "checkpoints_by_iteration": by_iteration,
                "completed_refinement_iterations": refinement_iteration,
            }
            self._save_json(output_name, payload)

        save_progress()

        stop = False
        for pass_index in range(1, max_pass_needed + 1):
            order = list(range(ntrain))
            rng.shuffle(order)
            for camera_index in order:
                if refinement_iteration >= stop_iterations:
                    stop = True
                    break

                refinement_iteration += 1
                self.global_iteration += 1
                self.gaussians.update_learning_rate(refinement_iteration)

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
                    self.gaussians.optimizer.step()
                    self.gaussians.optimizer.zero_grad(set_to_none=True)

                if refinement_iteration in iter_checkpoints:
                    item = evaluate_current(f"iter{refinement_iteration}")
                    item["requested_iteration_checkpoint"] = refinement_iteration
                    by_iteration[str(refinement_iteration)] = item
                    save_progress()

            full_pass_completed = refinement_iteration == pass_index * ntrain
            if full_pass_completed:
                print(
                    f"[final-quality] completed pass {pass_index} "
                    f"({refinement_iteration} refinement iterations)",
                    flush=True,
                )
                if pass_index in pass_checkpoints:
                    item = evaluate_current(f"pass{pass_index}")
                    item["pass"] = pass_index
                    by_pass[str(pass_index)] = item
                    save_progress()
            if stop or refinement_iteration >= stop_iterations:
                break

        # stop_iterations is always itself the maximum requested pass endpoint or
        # exact-iteration endpoint, so metrics should already exist. Be defensive.
        final_item = metrics_cache.get(refinement_iteration)
        if final_item is None:
            final_item = evaluate_current(f"final_iter{refinement_iteration}")

        endpoint_metrics = {
            "protocol": "online endpoint + one continuous post-hoc trajectory with pass and exact-update checkpoints",
            "checkpoint_passes": list(pass_checkpoints),
            "checkpoint_iterations": list(iter_checkpoints),
            "num_train_views": ntrain,
            "num_test_views": ntest,
            "stop_refinement_iterations": refinement_iteration,
            "stop_equivalent_passes": refinement_iteration / ntrain,
            "continue_online_optimizer": True,
            "restart_refinement_xyz_lr_index": True,
            "opacity_reset": reset_stats,
            "topology_fixed": True,
            "densification": False,
            "pruning": False,
            "gaussians": g_before,
            "online": {"train": online_train, "test": online_test},
            "checkpoints_by_pass": by_pass,
            "checkpoints_by_iteration": by_iteration,
        }
        self._save_json(output_name, endpoint_metrics)

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
                "train_inserted": final_item["train"],
                "test_all": final_item["test"],
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

    BackendEvaluationMixin._finalize_impl = finalize_with_dual_checkpoints


if __name__ == "__main__":
    _disable_intermediate_online_evaluation()
    _install_lpips_evaluator()
    _install_dual_checkpoint_refinement()
    safe.repro.main()

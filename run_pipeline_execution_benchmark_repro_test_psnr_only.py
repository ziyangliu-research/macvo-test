#!/usr/bin/env python3
"""Reproducible serial benchmark with final held-out Test PSNR only.

This wrapper keeps the online mapping algorithm unchanged and replaces the
normal finalize-time evaluation with a single metric: mean PSNR over all
held-out test cameras. It performs no post-hoc/global refinement and skips
train/active-map evaluation, SSIM, LPIPS, ATE, and image export.

Aggressive opacity pruning can temporarily prune the map to zero Gaussians.
The CUDA rasterizer's zero-Gaussian backward path returns an invalid SH-gradient
shape, so this wrapper skips only those meaningless backward passes until the
next train packet appends Gaussians again. This guard applies both with and
without historical replay. Pruning itself is not weakened and no Gaussian is
rescued.
"""
from __future__ import annotations

import os
import random
import time

import torch

import run_pipeline_execution_benchmark_repro_empty_safe as safe


def _install_no_replay_empty_map_guard() -> None:
    """Make the recent-only optimizer safe when maintenance empties the map."""
    from async_pipeline.backend_core import StreamingIncrementalBackend

    original_optimize = StreamingIncrementalBackend._optimize_active_map
    if getattr(original_optimize, "_no_replay_empty_map_guard", False):
        return

    def optimize_recent_only_empty_safe(self, update, active_cameras):
        from utils.loss_utils import l1_loss, ssim

        maintenance_event = None
        stack = list(active_cameras)
        empty_map_skipped_iterations = 0

        for local_iteration in range(1, self.config.iterations_per_packet + 1):
            self.global_iteration += 1
            self.gaussians.update_learning_rate(self.global_iteration)
            if not stack:
                stack = list(active_cameras)
            camera = stack.pop(random.randrange(len(stack)))

            # diff-gaussian-rasterization accepts an empty forward pass, but its
            # backward returns an SH gradient shaped [0,0,3] instead of [0,C,3].
            # Preserve the pruning result and simply skip backward until a later
            # train packet inserts Gaussians again.
            if int(self.gaussians.get_xyz.shape[0]) == 0:
                empty_map_skipped_iterations += 1
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                if empty_map_skipped_iterations == 1:
                    print(
                        "[empty-map guard/no-replay] "
                        f"packet={self.train_packet_count} "
                        f"frame={update.descriptor.frame_index} "
                        f"first_skipped_local_iter={local_iteration}; "
                        "maintenance pruned all Gaussians",
                        flush=True,
                    )
                continue

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
                collecting = (
                    self.config.maintenance_mode == "standard"
                    and local_iteration <= self.config.maintenance_after_local_iteration
                )
                if collecting:
                    indices = self._visibility_indices(
                        render_pkg["visibility_filter"],
                        int(self.gaussians.get_xyz.shape[0]),
                    )
                    radii = render_pkg["radii"]
                    if indices.numel() > 0:
                        self.gaussians.max_radii2D[indices] = torch.maximum(
                            self.gaussians.max_radii2D[indices], radii[indices]
                        )
                        self.gaussians.add_densification_stats(
                            render_pkg["viewspace_points"], indices
                        )

                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)

                if (
                    self.config.maintenance_mode == "standard"
                    and local_iteration == self.config.maintenance_after_local_iteration
                ):
                    maintenance_event = self._run_maintenance(
                        update,
                        local_iteration,
                        render_pkg["radii"],
                    )

                if (
                    self.wandb_run is not None
                    and self.global_iteration % self.config.wandb_log_interval == 0
                ):
                    self.wandb_run.log(
                        {
                            "train/loss": float(loss.item()),
                            "train/l1": float(ll1.item()),
                            "train/ssim": float(ssim_value.item()),
                            "scene/num_gaussians": int(self.gaussians.get_xyz.shape[0]),
                            "stream/frame_index": update.descriptor.frame_index,
                            "stream/train_packet_count": self.train_packet_count,
                        },
                        step=self.global_iteration,
                    )

        return maintenance_event

    optimize_recent_only_empty_safe._no_replay_empty_map_guard = True  # type: ignore[attr-defined]
    StreamingIncrementalBackend._optimize_active_map = optimize_recent_only_empty_safe
    print(
        "[repro] recent-only optimizer enabled with zero-Gaussian backward guard",
        flush=True,
    )


def _install_test_psnr_only_finalize() -> None:
    from async_pipeline.backend_evaluation import BackendEvaluationMixin

    if getattr(BackendEvaluationMixin._finalize_impl, "_test_psnr_only_patch", False):
        return

    @torch.no_grad()
    def finalize_test_psnr_only(self):
        self.initialize()

        from utils.image_utils import psnr

        values = []
        if self.gaussians is not None and int(self.gaussians.get_xyz.shape[0]) > 0:
            for camera in self.test_cameras:
                image = self.render(
                    camera,
                    self.gaussians,
                    self.pipe,
                    self.background,
                    use_trained_exp=False,
                    separate_sh=False,
                )["render"].clamp(0.0, 1.0)
                gt = camera.original_image.clamp(0.0, 1.0)
                values.append(float(psnr(image, gt).mean().item()))

        test_metrics = {
            "num_views": len(values),
            "psnr": (sum(values) / len(values)) if values else None,
        }
        payload = {
            "protocol": "online endpoint only; strict held-out Test PSNR",
            "num_train_packets": int(self.train_packet_count),
            "num_test_views": len(self.test_cameras),
            "global_iteration": int(self.global_iteration),
            "num_gaussians": (
                0 if self.gaussians is None else int(self.gaussians.get_xyz.shape[0])
            ),
            "test": test_metrics,
        }
        self._save_json("test_psnr_only.json", payload)

        psnr_value = test_metrics["psnr"]
        if psnr_value is None:
            print("[test-psnr-only] Test PSNR=MISSING (0 views)", flush=True)
        else:
            print(
                f"[test-psnr-only] views={len(values)} PSNR={psnr_value:.6f} dB "
                f"G={payload['num_gaussians']}",
                flush=True,
            )

        summary = {
            "backend": "StreamingIncrementalBackend",
            "num_train_packets": int(self.train_packet_count),
            "num_train_cameras": len(self.train_cameras),
            "num_test_cameras": len(self.test_cameras),
            "total_iterations": int(self.global_iteration),
            "final_num_gaussians": payload["num_gaussians"],
            "final_metrics": {"test_all": test_metrics},
            "gpu_memory": self._gpu_memory_stats(),
            "wall_time_sec": time.perf_counter() - self.wall_start,
        }
        self._save_json("incremental_backend_summary.json", summary)
        if self.wandb_run is not None:
            self.wandb_run.finish()
        return summary

    finalize_test_psnr_only._test_psnr_only_patch = True  # type: ignore[attr-defined]
    BackendEvaluationMixin._finalize_impl = finalize_test_psnr_only


if __name__ == "__main__":
    _install_test_psnr_only_finalize()

    replay_raw = os.environ.get("PIPELINE_HISTORICAL_REPLAY_FRACTION")
    replay_fraction = float(replay_raw) if replay_raw is not None else 0.0
    if replay_fraction <= 0.0:
        _install_no_replay_empty_map_guard()

    safe.repro.main()

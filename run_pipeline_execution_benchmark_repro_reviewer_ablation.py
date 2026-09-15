#!/usr/bin/env python3
"""Reviewer ablation runner for SH003 stage order and insertion opacity cap.

This wrapper keeps the formal serial MAC-VO + ReSplat + incremental GraphDECO
pipeline unchanged except for an opt-in historical replay ordering variable:

  PIPELINE_HISTORICAL_REPLAY_ORDER=post_maintenance  (formal method)
  PIPELINE_HISTORICAL_REPLAY_ORDER=from_start        (stage-order ablation)

For B=100 and rho=0.30, both modes execute exactly 70 recent-view and 30
historical-view optimizer updates whenever a historical pool exists.  The only
change is where those 30 historical updates occur:

  post_maintenance: iterations 1..50 recent-only; 30 history slots distributed
                    across iterations 51..100.
  from_start:       30 history slots distributed across iterations 1..100.

Maintenance remains at the configured local iteration (M=50 in the paper
protocol), so the from-start variant intentionally lets historical supervision
contribute to pre-maintenance optimization/densification statistics.  This is
the controlled variable requested by the stage-order ablation.

Intermediate metric rendering is disabled.  Final Train/Test PSNR/SSIM remains
enabled in backend.finalize(), while frame_timing_log.json therefore measures
online processing only.  A zero-Gaussian guard is retained for robustness under
aggressive pruning; it does not rescue Gaussians or weaken the threshold.
"""
from __future__ import annotations

import os
import random

import torch

import run_pipeline_execution_benchmark_repro as repro


def _install_exact_count_ordered_replay() -> None:
    from async_pipeline.backend_core import StreamingIncrementalBackend

    original_optimize = StreamingIncrementalBackend._optimize_active_map
    if getattr(original_optimize, "_reviewer_ordered_replay_patch", False):
        return

    replay_fraction = float(os.environ.get("PIPELINE_HISTORICAL_REPLAY_FRACTION", "0"))
    if not 0.0 < replay_fraction < 1.0:
        raise ValueError(
            "PIPELINE_HISTORICAL_REPLAY_FRACTION must be strictly between 0 and 1"
        )

    order_mode = os.environ.get(
        "PIPELINE_HISTORICAL_REPLAY_ORDER", "post_maintenance"
    ).strip().lower()
    if order_mode not in {"post_maintenance", "from_start"}:
        raise ValueError(
            "PIPELINE_HISTORICAL_REPLAY_ORDER must be post_maintenance or from_start"
        )

    def optimize_with_ordered_replay(self, update, active_cameras):
        from utils.loss_utils import l1_loss, ssim

        maintenance_event = None
        recent_pool = list(active_cameras)
        recent_stack = list(recent_pool)
        historical_pool = list(self.train_cameras[: -self.config.local_map_size])
        historical_stack = list(historical_pool)

        total_iterations = int(self.config.iterations_per_packet)
        if order_mode == "from_start":
            replay_start = 1
        elif self.config.maintenance_mode == "standard":
            replay_start = int(self.config.maintenance_after_local_iteration) + 1
        else:
            replay_start = 1
        replay_window = max(0, total_iterations - replay_start + 1)

        requested_history = int(round(total_iterations * replay_fraction))
        target_history = (
            min(replay_window, requested_history) if historical_pool else 0
        )
        recent_count = 0
        history_count = 0
        empty_map_skipped_iterations = 0

        for local_iteration in range(1, total_iterations + 1):
            self.global_iteration += 1
            self.gaussians.update_learning_rate(self.global_iteration)

            use_history = False
            if target_history > 0 and local_iteration >= replay_start:
                replay_index = local_iteration - replay_start + 1
                before = ((replay_index - 1) * target_history) // replay_window
                after = (replay_index * target_history) // replay_window
                use_history = after > before

            if use_history:
                if not historical_stack:
                    historical_stack = list(historical_pool)
                camera = historical_stack.pop(random.randrange(len(historical_stack)))
                history_count += 1
                supervision_source = "history"
            else:
                if not recent_stack:
                    recent_stack = list(recent_pool)
                camera = recent_stack.pop(random.randrange(len(recent_stack)))
                recent_count += 1
                supervision_source = "recent"

            background = (
                torch.rand(3, device=self.device)
                if self.opt.random_background
                else self.background
            )

            # Preserve the existing empty-map robustness without altering the
            # ablation itself.  No Gaussian is rescued and no threshold changes.
            if int(self.gaussians.get_xyz.shape[0]) == 0:
                empty_map_skipped_iterations += 1
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                if empty_map_skipped_iterations == 1:
                    print(
                        "[empty-map guard] "
                        f"packet={self.train_packet_count} "
                        f"frame={update.descriptor.frame_index} "
                        f"first_skipped_local_iter={local_iteration}",
                        flush=True,
                    )
                continue

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
                    and local_iteration
                    <= self.config.maintenance_after_local_iteration
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
                    and local_iteration
                    == self.config.maintenance_after_local_iteration
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
                            "replay/is_history_iteration": int(
                                supervision_source == "history"
                            ),
                        },
                        step=self.global_iteration,
                    )

        replay_stats = {
            "enabled": bool(historical_pool),
            "order_mode": order_mode,
            "requested_fraction": replay_fraction,
            "requested_history_iterations": requested_history,
            "historical_pool_size": len(historical_pool),
            "recent_pool_size": len(recent_pool),
            "replay_start_local_iteration": replay_start,
            "recent_iterations": recent_count,
            "historical_iterations": history_count,
            "empty_map_skipped_iterations": empty_map_skipped_iterations,
            "total_iterations": total_iterations,
        }
        setattr(self, "_historical_replay_last_stats", replay_stats)
        print(
            "[historical replay/reviewer] "
            f"order={order_mode} packet={self.train_packet_count} "
            f"frame={update.descriptor.frame_index} "
            f"recent={recent_count} history={history_count} "
            f"history_pool={len(historical_pool)} start_iter={replay_start}",
            flush=True,
        )
        return maintenance_event

    optimize_with_ordered_replay._reviewer_ordered_replay_patch = True  # type: ignore[attr-defined]
    optimize_with_ordered_replay._historical_replay_patch = True  # type: ignore[attr-defined]
    StreamingIncrementalBackend._optimize_active_map = optimize_with_ordered_replay

    print(
        "[reviewer ablation] exact-count historical replay enabled: "
        f"order={order_mode}, fraction={replay_fraction}",
        flush=True,
    )


def _disable_intermediate_evaluation() -> None:
    from async_pipeline.backend_evaluation import BackendEvaluationMixin

    original = BackendEvaluationMixin._record_evaluation
    if getattr(original, "_reviewer_final_eval_only_patch", False):
        return

    def no_intermediate_evaluation(self, stage, update, active_cameras):
        return None

    no_intermediate_evaluation._reviewer_final_eval_only_patch = True  # type: ignore[attr-defined]
    BackendEvaluationMixin._record_evaluation = no_intermediate_evaluation
    print(
        "[reviewer ablation] intermediate metric rendering disabled; "
        "final Train/Test evaluation preserved",
        flush=True,
    )


# Replace only the replay installer.  repro.main() still owns seeding, default
# CUDA-stream pinning, system construction, and the serial benchmark runner.
repro._install_historical_replay = _install_exact_count_ordered_replay


if __name__ == "__main__":
    _disable_intermediate_evaluation()
    repro.main()

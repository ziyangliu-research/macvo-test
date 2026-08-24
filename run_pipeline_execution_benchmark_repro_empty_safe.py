#!/usr/bin/env python3
"""Run the reproducible benchmark with an empty-map-safe replay optimizer.

This wrapper is intended for aggressive global-opacity pruning ablations.  It
keeps the normal replay path unchanged while Gaussians exist.  If maintenance
prunes the map to zero Gaussians, the remaining optimization slots in that
packet still advance the global iteration counter and replay schedule, but
render/backward is skipped until a later packet appends Gaussians again.

No minimum Gaussian count is enforced and no Gaussian is rescued from pruning,
so a threshold that empties the map is still represented faithfully in the
experiment rather than being silently weakened.
"""
from __future__ import annotations

import os
import random

import torch

import run_pipeline_execution_benchmark_repro as repro


def _install_historical_replay_empty_safe() -> None:
    """Install the existing fixed-budget replay with a zero-Gaussian guard."""

    from async_pipeline.backend_core import StreamingIncrementalBackend

    original_optimize = StreamingIncrementalBackend._optimize_active_map
    if getattr(original_optimize, "_historical_replay_patch", False):
        return

    replay_fraction = float(os.environ.get("PIPELINE_HISTORICAL_REPLAY_FRACTION", "0"))
    if not 0.0 < replay_fraction < 1.0:
        raise ValueError(
            "PIPELINE_HISTORICAL_REPLAY_FRACTION must be strictly between 0 and 1"
        )

    def optimize_with_historical_replay(
        self: StreamingIncrementalBackend,
        update,
        active_cameras,
    ):
        from utils.loss_utils import l1_loss, ssim

        maintenance_event = None
        recent_pool = list(active_cameras)
        recent_stack = list(recent_pool)
        historical_pool = list(self.train_cameras[: -self.config.local_map_size])
        historical_stack = list(historical_pool)

        total_iterations = int(self.config.iterations_per_packet)
        if self.config.maintenance_mode == "standard":
            replay_start = int(self.config.maintenance_after_local_iteration) + 1
        else:
            replay_start = 1
        replay_window = max(0, total_iterations - replay_start + 1)

        target_history = (
            min(
                replay_window,
                int(round(total_iterations * replay_fraction)),
            )
            if historical_pool
            else 0
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

            # diff-gaussian-rasterization can run a zero-Gaussian forward pass,
            # but its backward currently returns an SH gradient shaped [0,0,3]
            # instead of the model's [0,C,3].  Do not alter pruning to hide this
            # condition; simply skip the meaningless backward until the next
            # packet repopulates the map.
            if int(self.gaussians.get_xyz.shape[0]) == 0:
                empty_map_skipped_iterations += 1
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                if empty_map_skipped_iterations == 1:
                    print(
                        "[empty-map guard] "
                        f"packet={self.train_packet_count} "
                        f"frame={update.descriptor.frame_index} "
                        f"first_skipped_local_iter={local_iteration}; "
                        "maintenance pruned all Gaussians; remaining backward passes "
                        "are skipped until a later packet appends new Gaussians",
                        flush=True,
                    )
                if (
                    self.wandb_run is not None
                    and self.global_iteration % self.config.wandb_log_interval == 0
                ):
                    self.wandb_run.log(
                        {
                            "scene/num_gaussians": 0,
                            "stream/frame_index": update.descriptor.frame_index,
                            "stream/train_packet_count": self.train_packet_count,
                            "replay/is_history_iteration": int(
                                supervision_source == "history"
                            ),
                            "replay/empty_map_skip": 1,
                        },
                        step=self.global_iteration,
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
                            "scene/num_gaussians": int(
                                self.gaussians.get_xyz.shape[0]
                            ),
                            "stream/frame_index": update.descriptor.frame_index,
                            "stream/train_packet_count": self.train_packet_count,
                            "replay/is_history_iteration": int(
                                supervision_source == "history"
                            ),
                            "replay/empty_map_skip": 0,
                        },
                        step=self.global_iteration,
                    )

        replay_stats = {
            "enabled": bool(historical_pool),
            "requested_fraction": replay_fraction,
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
            "[historical replay] "
            f"packet={self.train_packet_count} "
            f"frame={update.descriptor.frame_index} "
            f"recent={recent_count} history={history_count} "
            f"history_pool={len(historical_pool)} "
            f"start_iter={replay_start} "
            f"empty_skips={empty_map_skipped_iterations}",
            flush=True,
        )
        return maintenance_event

    optimize_with_historical_replay._historical_replay_patch = True  # type: ignore[attr-defined]
    StreamingIncrementalBackend._optimize_active_map = optimize_with_historical_replay

    print(
        "[repro] fixed-budget historical replay enabled (empty-map safe): "
        f"fraction={replay_fraction}; pre-maintenance optimization remains recent-only",
        flush=True,
    )


# Preserve every other reproducibility/diagnostic behavior in the original
# wrapper; replace only the replay installer before its main() dispatches.
repro._install_historical_replay = _install_historical_replay_empty_safe


if __name__ == "__main__":
    repro.main()

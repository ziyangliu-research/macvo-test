#!/usr/bin/env python3
"""Run the reproducible no-replay benchmark with a zero-Gaussian guard.

This wrapper is for aggressive global opacity-pruning ablations with historical
replay disabled.  While the map is non-empty it reproduces the stock
StreamingIncrementalBackend._optimize_active_map path.  If maintenance prunes
all Gaussians, the remaining optimization slots for that packet still advance
iteration/camera/background sampling, but render/backward is skipped until a
later packet appends Gaussians again.

No Gaussian is rescued and the configured pruning threshold is never changed.
"""
from __future__ import annotations

import os
import random
from typing import Any, Optional, Sequence

import torch

import run_pipeline_execution_benchmark_repro as repro


def _install_no_replay_empty_safe() -> None:
    from async_pipeline.backend_camera import StreamingCamera
    from async_pipeline.backend_core import StreamingIncrementalBackend

    original_optimize = StreamingIncrementalBackend._optimize_active_map
    if getattr(original_optimize, "_no_replay_empty_safe_patch", False):
        return

    def optimize_no_replay_empty_safe(
        self: StreamingIncrementalBackend,
        update,
        active_cameras: Sequence[StreamingCamera],
    ) -> Optional[dict[str, Any]]:
        from utils.loss_utils import l1_loss, ssim

        maintenance_event: Optional[dict[str, Any]] = None
        stack = list(active_cameras)
        empty_map_skipped_iterations = 0

        for local_iteration in range(1, self.config.iterations_per_packet + 1):
            self.global_iteration += 1
            self.gaussians.update_learning_rate(self.global_iteration)

            if not stack:
                stack = list(active_cameras)
            camera = stack.pop(random.randrange(len(stack)))
            background = (
                torch.rand(3, device=self.device)
                if self.opt.random_background
                else self.background
            )

            # Preserve the configured aggressive prune exactly.  GraphDECO's
            # zero-Gaussian forward is valid, but the rasterizer backward returns
            # an incompatible SH gradient shape, so skip only the meaningless
            # backward iterations after the map has actually become empty.
            if int(self.gaussians.get_xyz.shape[0]) == 0:
                empty_map_skipped_iterations += 1
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                if empty_map_skipped_iterations == 1:
                    print(
                        "[empty-map guard] "
                        f"packet={self.train_packet_count} "
                        f"frame={update.descriptor.frame_index} "
                        f"first_skipped_local_iter={local_iteration}; "
                        "maintenance pruned all Gaussians; remaining backward "
                        "passes are skipped until a later packet appends new Gaussians",
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
                            "scene/num_gaussians": int(
                                self.gaussians.get_xyz.shape[0]
                            ),
                            "stream/frame_index": update.descriptor.frame_index,
                            "stream/train_packet_count": self.train_packet_count,
                        },
                        step=self.global_iteration,
                    )

        if empty_map_skipped_iterations:
            print(
                "[no-replay empty-map summary] "
                f"packet={self.train_packet_count} "
                f"frame={update.descriptor.frame_index} "
                f"empty_skips={empty_map_skipped_iterations}",
                flush=True,
            )
        return maintenance_event

    optimize_no_replay_empty_safe._no_replay_empty_safe_patch = True  # type: ignore[attr-defined]
    StreamingIncrementalBackend._optimize_active_map = optimize_no_replay_empty_safe
    print("[repro] no-replay empty-map guard enabled", flush=True)


if __name__ == "__main__":
    replay_raw = os.environ.get("PIPELINE_HISTORICAL_REPLAY_FRACTION")
    if replay_raw is not None and float(replay_raw) > 0.0:
        raise RuntimeError(
            "This runner is specifically for replay=0. Unset "
            "PIPELINE_HISTORICAL_REPLAY_FRACTION."
        )
    _install_no_replay_empty_safe()
    repro.main()

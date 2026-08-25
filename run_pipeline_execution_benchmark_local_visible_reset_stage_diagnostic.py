#!/usr/bin/env python3
"""Stage-wise diagnosis for the local-visible opacity-reset experiment.

This wrapper keeps the local-visible reset experiment itself unchanged and adds
read-only evaluations at one target train packet (default: packet 40):

S0: after current packet append / camera insertion, before local-visible reset
S1: immediately after local-visible opacity reset, before optimization
S2: after the maintenance-boundary optimizer step, before maintenance
S3: immediately after maintenance
S4: after all per-packet optimization iterations, including historical replay

The fixed evaluation sets are:
- early_history: first N inserted train cameras (default N=10)
- recent_local: current active local window
- all_inserted: all train cameras inserted up to the target packet

No stage evaluation changes parameters, optimizer state, pruning, reset policy,
maintenance timing, historical-replay timing, random sampling, or iteration
budget.  Evaluation uses BackendEvaluationMixin._evaluate under torch.no_grad().

Environment variables:
PIPELINE_HISTORICAL_REPLAY_FRACTION
    Required.  Normally 0.2.
PIPELINE_LOCAL_VISIBLE_RESET_MAX_OPACITY
    Optional local-visible reset cap; defaults to backend reset cap.
PIPELINE_LOCAL_RESET_DIAGNOSTIC_PACKET
    Target train packet count; defaults to 40.
PIPELINE_LOCAL_RESET_DIAGNOSTIC_EARLY_VIEWS
    Number of earliest cameras in the early-history fixed set; defaults to 10.
"""
from __future__ import annotations

import os
import random
from typing import Any, Sequence

import torch

import run_pipeline_execution_benchmark_repro as repro
from run_pipeline_execution_benchmark_local_visible_opacity_reset import (
    _reset_local_visible_opacity,
)


def _install_local_visible_reset_stage_diagnostic() -> None:
    from async_pipeline.backend_core import StreamingIncrementalBackend

    replay_fraction = float(
        os.environ.get("PIPELINE_HISTORICAL_REPLAY_FRACTION", "0")
    )
    if not 0.0 < replay_fraction < 1.0:
        raise ValueError(
            "PIPELINE_HISTORICAL_REPLAY_FRACTION must be strictly between 0 and 1"
        )

    target_packet = int(
        os.environ.get("PIPELINE_LOCAL_RESET_DIAGNOSTIC_PACKET", "40")
    )
    early_views = int(
        os.environ.get("PIPELINE_LOCAL_RESET_DIAGNOSTIC_EARLY_VIEWS", "10")
    )
    if target_packet <= 0:
        raise ValueError("diagnostic packet must be positive")
    if early_views <= 0:
        raise ValueError("diagnostic early-view count must be positive")

    def record_stage(
        self: StreamingIncrementalBackend,
        update: Any,
        active_cameras: Sequence[Any],
        stage: str,
        local_iteration: int,
    ) -> None:
        early = list(self.train_cameras[:early_views])
        recent = list(active_cameras)
        all_inserted = list(self.train_cameras)

        entry: dict[str, Any] = {
            "stage": stage,
            "frame_index": int(update.descriptor.frame_index),
            "train_packet_count": int(self.train_packet_count),
            "global_iteration": int(self.global_iteration),
            "local_iteration": int(local_iteration),
            "num_gaussians": int(self.gaussians.get_xyz.shape[0]),
            "early_history": self._evaluate(early),
            "recent_local": self._evaluate(recent),
            "all_inserted": self._evaluate(all_inserted),
        }

        log = getattr(self, "_local_reset_stage_diagnostic_log", None)
        if log is None:
            log = []
            setattr(self, "_local_reset_stage_diagnostic_log", log)
        log.append(entry)
        if self.config.write_runtime_artifacts:
            self._save_json("local_reset_stage_diagnostic_log.json", log)

        def p(split: str) -> float:
            return float(entry[split].get("psnr", float("nan")))

        print(
            "[local-reset stage diagnostic] "
            f"{stage} packet={self.train_packet_count} "
            f"frame={update.descriptor.frame_index} "
            f"local_iter={local_iteration} G={entry['num_gaussians']} "
            f"early={p('early_history'):.4f} "
            f"recent={p('recent_local'):.4f} "
            f"all={p('all_inserted'):.4f}",
            flush=True,
        )

    def optimize_with_diagnostics(
        self: StreamingIncrementalBackend,
        update: Any,
        active_cameras: Sequence[Any],
    ):
        from utils.loss_utils import l1_loss, ssim

        is_target = int(self.train_packet_count) == target_packet
        if is_target:
            record_stage(self, update, active_cameras, "S0_pre_reset", 0)

        # This is exactly the experimental local-visible reset used by the
        # non-diagnostic wrapper.  Its own reset log is still written normally.
        _reset_local_visible_opacity(self, update, active_cameras)

        if is_target:
            record_stage(self, update, active_cameras, "S1_post_reset", 0)

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

                maintenance_boundary = (
                    self.config.maintenance_mode == "standard"
                    and local_iteration
                    == self.config.maintenance_after_local_iteration
                )

                if is_target and maintenance_boundary:
                    record_stage(
                        self,
                        update,
                        active_cameras,
                        "S2_pre_maintenance",
                        local_iteration,
                    )

                if maintenance_boundary:
                    maintenance_event = self._run_maintenance(
                        update,
                        local_iteration,
                        render_pkg["radii"],
                    )
                    if is_target:
                        record_stage(
                            self,
                            update,
                            active_cameras,
                            "S3_post_maintenance",
                            local_iteration,
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

        if is_target:
            record_stage(
                self,
                update,
                active_cameras,
                "S4_post_optimization",
                total_iterations,
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

    optimize_with_diagnostics._historical_replay_patch = True  # type: ignore[attr-defined]
    optimize_with_diagnostics._local_visible_opacity_reset_patch = True  # type: ignore[attr-defined]
    optimize_with_diagnostics._local_reset_stage_diagnostic_patch = True  # type: ignore[attr-defined]
    StreamingIncrementalBackend._optimize_active_map = optimize_with_diagnostics

    print(
        "[experiment] local-visible opacity reset + packet-stage diagnostic enabled; "
        f"target_packet={target_packet}, early_views={early_views}, "
        f"replay_fraction={replay_fraction}",
        flush=True,
    )


# Override any installer side effects introduced by importing the local-reset
# wrapper.  repro.main() will call this composed diagnostic implementation.
repro._install_historical_replay = _install_local_visible_reset_stage_diagnostic


if __name__ == "__main__":
    _raw = os.environ.get("PIPELINE_HISTORICAL_REPLAY_FRACTION")
    if _raw is None or float(_raw) <= 0.0:
        raise RuntimeError(
            "Set PIPELINE_HISTORICAL_REPLAY_FRACTION, normally 0.2, for this experiment."
        )
    repro.main()

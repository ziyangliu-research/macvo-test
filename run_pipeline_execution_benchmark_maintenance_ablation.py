#!/usr/bin/env python3
"""Run the reproducible pipeline with a maintenance-component ablation.

This wrapper is intentionally narrow: it patches only
``StreamingIncrementalBackend._run_maintenance`` and then delegates to
``run_pipeline_execution_benchmark_repro.py``. All existing reproducibility,
serial ReSplat stream handling, historical replay, and root-cause diagnostics
therefore remain available.

Select the maintenance operation with ``PIPELINE_MAINTENANCE_ABLATION_MODE``:

- ``standard``: use the unmodified backend maintenance;
- ``densify_only``: run GraphDECO clone + split, but skip the final opacity /
  size pruning step. Note that split necessarily replaces selected parent
  Gaussians with children, exactly as GraphDECO's split primitive does;
- ``prune_only``: skip clone/split and apply only the pruning mask that the
  standard backend would apply after densification.

Keep ``backend.maintenance_mode=standard`` in the pipeline config. That setting
is still needed so the optimizer collects densification statistics and invokes
maintenance at the configured local iteration.
"""
from __future__ import annotations

import os
import time
from typing import Any

import torch


def _install_maintenance_ablation(mode: str) -> None:
    from async_pipeline.backend_core import StreamingIncrementalBackend

    mode = mode.strip().lower()
    if mode not in {"standard", "densify_only", "prune_only"}:
        raise ValueError(
            "PIPELINE_MAINTENANCE_ABLATION_MODE must be one of "
            "standard, densify_only, prune_only"
        )
    if mode == "standard":
        print("[maintenance ablation] standard backend maintenance", flush=True)
        return

    original = StreamingIncrementalBackend._run_maintenance
    if getattr(original, "_maintenance_component_ablation_patch", False):
        return

    def run_maintenance_component_ablation(
        self: StreamingIncrementalBackend,
        update,
        local_iteration: int,
        radii: torch.Tensor,
    ) -> dict[str, Any]:
        g = self.gaussians
        count_before = int(g.get_xyz.shape[0])
        threshold = (
            self.config.maintenance_grad_threshold
            if self.config.maintenance_grad_threshold > 0
            else self.opt.densify_grad_threshold
        )
        max_screen = (
            self.config.maintenance_max_screen_size
            if self.config.maintenance_max_screen_size > 0
            else None
        )
        extent = self._current_scene_extent()

        details: dict[str, Any] = {"ablation_mode": mode}
        start = time.perf_counter()

        if mode == "densify_only":
            # Match the densification half of GraphDECO densify_and_prune().
            # Keep the same pre-maintenance gradient statistics and radii.
            grads = g.xyz_gradient_accum / g.denom
            grads[grads.isnan()] = 0.0
            g.tmp_radii = radii

            before_clone = int(g.get_xyz.shape[0])
            g.densify_and_clone(grads, threshold, extent)
            after_clone = int(g.get_xyz.shape[0])
            g.densify_and_split(grads, threshold, extent)
            after_split = int(g.get_xyz.shape[0])

            # GraphDECO clears this temporary field at the end of
            # densify_and_prune(); do the same here.
            g.tmp_radii = None
            torch.cuda.empty_cache()

            details.update(
                {
                    "clone_added": after_clone - before_clone,
                    # For GraphDECO's N=2 split this is the net count increase
                    # after replacing each selected parent by two children.
                    "split_net_added": after_split - after_clone,
                    "opacity_pruned": 0,
                    "total_pruned": 0,
                }
            )

        elif mode == "prune_only":
            # Match the pruning half of GraphDECO densify_and_prune(), but on
            # the current pre-densification Gaussian set.
            #
            # prune_points() also slices GraphDECO's temporary tmp_radii buffer,
            # so standard densify_and_prune() always creates it before pruning.
            # Even though prune_only skips clone/split, we must provide the same
            # temporary state to keep prune_points() internally consistent.
            g.tmp_radii = radii

            opacity_mask = (
                g.get_opacity < self.config.maintenance_min_opacity
            ).squeeze()
            prune_mask = opacity_mask.clone()
            opacity_pruned = int(opacity_mask.sum().item())
            screen_or_world_pruned = 0

            if max_screen:
                big_points_vs = g.max_radii2D > max_screen
                big_points_ws = g.get_scaling.max(dim=1).values > 0.1 * extent
                size_mask = torch.logical_or(big_points_vs, big_points_ws)
                screen_or_world_pruned = int(
                    torch.logical_and(size_mask, ~opacity_mask).sum().item()
                )
                prune_mask = torch.logical_or(prune_mask, size_mask)

            total_pruned = int(prune_mask.sum().item())
            g.prune_points(prune_mask)

            # Match GraphDECO densify_and_prune() cleanup after prune_points().
            g.tmp_radii = None
            torch.cuda.empty_cache()

            details.update(
                {
                    "clone_added": 0,
                    "split_net_added": 0,
                    "opacity_pruned": opacity_pruned,
                    "screen_or_world_pruned_exclusive": screen_or_world_pruned,
                    "total_pruned": total_pruned,
                }
            )

        sec = self._sync_elapsed(start)
        event: dict[str, Any] = {
            "frame_index": update.descriptor.frame_index,
            "global_iteration": self.global_iteration,
            "local_iteration": local_iteration,
            "count_before": count_before,
            "count_after": int(g.get_xyz.shape[0]),
            "grad_threshold": threshold,
            "min_opacity": self.config.maintenance_min_opacity,
            "max_screen_size": max_screen,
            "scene_extent": extent,
            "maintenance_sec": sec,
            **details,
        }
        self.maintenance_log.append(event)
        if self.config.write_runtime_artifacts:
            self._save_json("maintenance_log.json", self.maintenance_log)

        print(
            "[maintenance ablation] "
            f"mode={mode} frame={update.descriptor.frame_index} "
            f"G={count_before}->{event['count_after']} "
            f"clone_added={event['clone_added']} "
            f"split_net_added={event['split_net_added']} "
            f"opacity_pruned={event['opacity_pruned']} "
            f"total_pruned={event['total_pruned']}",
            flush=True,
        )
        return event

    run_maintenance_component_ablation._maintenance_component_ablation_patch = True  # type: ignore[attr-defined]
    StreamingIncrementalBackend._run_maintenance = run_maintenance_component_ablation
    print(f"[maintenance ablation] installed mode={mode}", flush=True)


def main() -> None:
    mode = os.environ.get("PIPELINE_MAINTENANCE_ABLATION_MODE", "standard")
    _install_maintenance_ablation(mode)

    # Import after patch installation so the reproducibility wrapper can layer
    # historical replay and S0-S4 diagnostics around this maintenance function.
    import run_pipeline_execution_benchmark_repro as repro

    repro.main()


if __name__ == "__main__":
    main()

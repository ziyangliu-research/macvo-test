#!/usr/bin/env python3
"""Run the reproducible pipeline with cohort-aware opacity pruning.

The standard incremental backend applies one global opacity threshold to the
entire cumulative Gaussian map.  That rule is inherited from GraphDECO, but it
is poorly matched to the current ReSplat initialization: every new packet is
capped/reset to opacity 0.01, while standard pruning removes only opacity <
0.005.  A Gaussian can therefore fail to grow beyond its reset level and still
survive indefinitely.

This ablation keeps the standard maintenance order and densification behavior:

    collect stats -> clone -> split -> prune

but uses two opacity thresholds at the final prune step:

- historical Gaussians: backend.maintenance_min_opacity (normally 0.005)
- current-packet Gaussians and their clone/split descendants:
  PIPELINE_NEW_PACKET_PRUNE_THRESHOLD (default 0.01)

Clone/split descendants inherit the cohort of their parent, so the final prune
remains cohort-aware even after structural densification.  The total
optimization budget, maintenance timing, historical replay, and opacity reset
policy are unchanged.

Results are written to ``cohort_pruning_log.json`` in the backend output
folder.  Keep ``backend.maintenance_mode=standard`` so pre-maintenance
statistics are collected and maintenance is invoked at the configured local
iteration.
"""
from __future__ import annotations

import os
import time
from typing import Any

import torch


def _install_cohort_pruning() -> None:
    from async_pipeline.backend_core import StreamingIncrementalBackend

    original_initialize = StreamingIncrementalBackend._initialize_gaussians
    original_append = StreamingIncrementalBackend._append_gaussians
    original_maintenance = StreamingIncrementalBackend._run_maintenance

    if getattr(original_maintenance, "_cohort_pruning_patch", False):
        return

    new_threshold = float(os.environ.get("PIPELINE_NEW_PACKET_PRUNE_THRESHOLD", "0.01"))
    if new_threshold <= 0.0 or new_threshold >= 1.0:
        raise ValueError("PIPELINE_NEW_PACKET_PRUNE_THRESHOLD must be in (0, 1)")

    def initialize_with_packet_count(
        self: StreamingIncrementalBackend,
        tensors: dict[str, torch.Tensor],
    ) -> None:
        setattr(self, "_cohort_current_packet_count", int(tensors["xyz"].shape[0]))
        original_initialize(self, tensors)

    def append_with_packet_count(
        self: StreamingIncrementalBackend,
        tensors: dict[str, torch.Tensor],
    ) -> None:
        setattr(self, "_cohort_current_packet_count", int(tensors["xyz"].shape[0]))
        original_append(self, tensors)

    def maintenance_with_cohort_pruning(
        self: StreamingIncrementalBackend,
        update,
        local_iteration: int,
        radii: torch.Tensor,
    ) -> dict[str, Any]:
        g = self.gaussians
        count_before = int(g.get_xyz.shape[0])
        current_count = int(getattr(self, "_cohort_current_packet_count", 0))
        current_count = min(max(current_count, 0), count_before)
        historical_count = count_before - current_count

        historical_threshold = float(self.config.maintenance_min_opacity)
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

        # Cohort mask before densification.  True means current packet.
        cohort = torch.zeros(count_before, dtype=torch.bool, device=self.device)
        if current_count > 0:
            cohort[historical_count:] = True

        grads = g.xyz_gradient_accum / g.denom
        grads[grads.isnan()] = 0.0
        g.tmp_radii = radii

        start = time.perf_counter()

        # ---- Clone: reproduce GraphDECO selection so cohort identity can be
        # propagated to appended clones.
        before_clone = int(g.get_xyz.shape[0])
        clone_selected = torch.where(
            torch.norm(grads, dim=-1) >= threshold,
            True,
            False,
        )
        clone_selected = torch.logical_and(
            clone_selected,
            torch.max(g.get_scaling, dim=1).values <= g.percent_dense * extent,
        )
        clone_cohort = cohort[clone_selected]
        g.densify_and_clone(grads, threshold, extent)
        after_clone = int(g.get_xyz.shape[0])
        if after_clone - before_clone != int(clone_cohort.numel()):
            raise RuntimeError(
                "cohort clone bookkeeping mismatch: "
                f"added={after_clone - before_clone} expected={clone_cohort.numel()}"
            )
        cohort = torch.cat([cohort, clone_cohort], dim=0)

        # ---- Split: GraphDECO evaluates the original gradient tensor padded
        # with zeros for clone rows, then replaces selected parents by N=2
        # children.  Reproduce that mask to propagate cohort identity.
        n_split_input = int(g.get_xyz.shape[0])
        padded_grad = torch.zeros(n_split_input, device=self.device)
        padded_grad[: grads.shape[0]] = grads.squeeze()
        split_selected = torch.where(padded_grad >= threshold, True, False)
        split_selected = torch.logical_and(
            split_selected,
            torch.max(g.get_scaling, dim=1).values > g.percent_dense * extent,
        )
        split_parent_cohort = cohort[split_selected]
        split_parent_count = int(split_selected.sum().item())
        g.densify_and_split(grads, threshold, extent)
        after_split = int(g.get_xyz.shape[0])

        # For N=2, GraphDECO removes each selected parent and appends two
        # children, so net count increase equals the number of split parents.
        expected_after_split = n_split_input + split_parent_count
        if after_split != expected_after_split:
            raise RuntimeError(
                "cohort split bookkeeping mismatch: "
                f"after={after_split} expected={expected_after_split}"
            )
        cohort = torch.cat(
            [cohort[~split_selected], split_parent_cohort.repeat(2)],
            dim=0,
        )
        if int(cohort.numel()) != after_split:
            raise RuntimeError(
                f"cohort mask size {cohort.numel()} != Gaussian count {after_split}"
            )

        # ---- Final opacity prune with cohort-specific thresholds.
        opacity = g.get_opacity.reshape(-1)
        historical_mask = ~cohort
        current_mask = cohort
        historical_prune = torch.logical_and(
            historical_mask,
            opacity < historical_threshold,
        )
        current_prune = torch.logical_and(
            current_mask,
            opacity < new_threshold,
        )
        prune_mask = torch.logical_or(historical_prune, current_prune)

        size_pruned_exclusive = 0
        if max_screen:
            big_points_vs = g.max_radii2D > max_screen
            big_points_ws = g.get_scaling.max(dim=1).values > 0.1 * extent
            size_mask = torch.logical_or(big_points_vs, big_points_ws)
            size_pruned_exclusive = int(
                torch.logical_and(size_mask, ~prune_mask).sum().item()
            )
            prune_mask = torch.logical_or(prune_mask, size_mask)

        historical_pruned = int(historical_prune.sum().item())
        current_pruned = int(current_prune.sum().item())
        current_before_prune = int(current_mask.sum().item())
        historical_before_prune = int(historical_mask.sum().item())
        total_pruned = int(prune_mask.sum().item())

        g.prune_points(prune_mask)
        g.tmp_radii = None
        torch.cuda.empty_cache()

        sec = self._sync_elapsed(start)
        count_after = int(g.get_xyz.shape[0])
        event: dict[str, Any] = {
            "frame_index": int(update.descriptor.frame_index),
            "train_packet_count": int(self.train_packet_count),
            "global_iteration": int(self.global_iteration),
            "local_iteration": int(local_iteration),
            "count_before": count_before,
            "count_after": count_after,
            "current_packet_count_pre_densify": current_count,
            "historical_count_pre_densify": historical_count,
            "clone_added": after_clone - before_clone,
            "split_parent_count": split_parent_count,
            "count_pre_final_prune": after_split,
            "current_count_pre_final_prune": current_before_prune,
            "historical_count_pre_final_prune": historical_before_prune,
            "new_packet_prune_threshold": new_threshold,
            "historical_prune_threshold": historical_threshold,
            "current_pruned": current_pruned,
            "historical_pruned": historical_pruned,
            "size_pruned_exclusive": size_pruned_exclusive,
            "total_pruned": total_pruned,
            "current_survived_final_prune": current_before_prune - current_pruned,
            "historical_survived_final_prune": historical_before_prune - historical_pruned,
            "grad_threshold": threshold,
            "max_screen_size": max_screen,
            "scene_extent": extent,
            "maintenance_sec": sec,
        }

        self.maintenance_log.append(event)
        log = getattr(self, "_cohort_pruning_log", None)
        if log is None:
            log = []
            setattr(self, "_cohort_pruning_log", log)
        log.append(event)

        if self.config.write_runtime_artifacts:
            self._save_json("maintenance_log.json", self.maintenance_log)
            self._save_json("cohort_pruning_log.json", log)

        print(
            "[cohort pruning] "
            f"packet={self.train_packet_count} frame={update.descriptor.frame_index} "
            f"G={count_before}->{count_after} "
            f"clone={after_clone - before_clone} split_parents={split_parent_count} "
            f"current={current_before_prune} prune={current_pruned} "
            f"survive={current_before_prune - current_pruned} "
            f"history_prune={historical_pruned} "
            f"new_thr={new_threshold} hist_thr={historical_threshold}",
            flush=True,
        )
        return event

    initialize_with_packet_count._cohort_pruning_patch = True  # type: ignore[attr-defined]
    append_with_packet_count._cohort_pruning_patch = True  # type: ignore[attr-defined]
    maintenance_with_cohort_pruning._cohort_pruning_patch = True  # type: ignore[attr-defined]

    StreamingIncrementalBackend._initialize_gaussians = initialize_with_packet_count
    StreamingIncrementalBackend._append_gaussians = append_with_packet_count
    StreamingIncrementalBackend._run_maintenance = maintenance_with_cohort_pruning

    print(
        "[cohort pruning] installed: "
        f"new_packet_threshold={new_threshold}, "
        "historical threshold remains backend.maintenance_min_opacity",
        flush=True,
    )


def main() -> None:
    _install_cohort_pruning()

    # Import after patch installation so fixed-budget historical replay and the
    # existing S0-S4 root-cause diagnostics wrap this maintenance unchanged.
    import run_pipeline_execution_benchmark_repro as repro

    repro.main()


if __name__ == "__main__":
    main()

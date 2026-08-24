#!/usr/bin/env python3
"""Run the reproducible pipeline with a fixed 8-recent + 2-history supervision window.

For each train packet, this wrapper constructs the optimization camera set as:

- the most recent 8 train cameras; plus
- up to 2 distinct cameras sampled uniformly from the older history pool.

Once at least 10 train cameras have been seen, the active optimization set has
exactly 10 views. The standard backend optimizer then samples without
replacement from this fixed 10-view set and refills the stack, so with the
canonical 100 iterations each of the 10 selected views is used exactly 10 times
per packet. Standard maintenance and pruning are unchanged.

The normal per-packet evaluation is also patched so ``active_local_map`` means
this exact 8+2 supervision set, while ``train_inserted`` continues to evaluate
all train cameras seen so far. This makes the W&B curves directly usable for a
recent-window versus global-history comparison.
"""
from __future__ import annotations

import random


def _install_fixed_8plus2() -> None:
    from async_pipeline.backend_core import StreamingIncrementalBackend

    original_optimize = StreamingIncrementalBackend._optimize_active_map
    original_record_evaluation = StreamingIncrementalBackend._record_evaluation

    if getattr(original_optimize, "_fixed_8plus2_patch", False):
        return

    recent_count = 8
    history_count = 2

    def optimize_fixed_8plus2(self: StreamingIncrementalBackend, update, active_cameras):
        # Ignore the caller-provided local window and construct the requested
        # fixed supervision set directly from all train cameras seen so far.
        recent = list(self.train_cameras[-recent_count:])
        historical_pool = list(self.train_cameras[:-recent_count])
        k = min(history_count, len(historical_pool))
        historical = random.sample(historical_pool, k=k) if k > 0 else []
        combined = recent + historical
        setattr(self, "_fixed_8plus2_active_cameras", combined)
        setattr(
            self,
            "_fixed_8plus2_last_stats",
            {
                "recent_views": len(recent),
                "historical_views": len(historical),
                "historical_pool_size": len(historical_pool),
                "total_active_views": len(combined),
                "historical_frame_indices": [int(c.frame_index) for c in historical],
            },
        )
        print(
            "[fixed 8+2] "
            f"packet={self.train_packet_count} frame={update.descriptor.frame_index} "
            f"recent={len(recent)} history={len(historical)} "
            f"history_pool={len(historical_pool)} "
            f"history_frames={[int(c.frame_index) for c in historical]}",
            flush=True,
        )
        return original_optimize(self, update, combined)

    def record_evaluation_fixed_8plus2(
        self: StreamingIncrementalBackend,
        stage: str,
        update,
        active_cameras,
    ) -> None:
        combined = getattr(self, "_fixed_8plus2_active_cameras", None)
        if combined is None:
            combined = active_cameras
        original_record_evaluation(self, stage, update, combined)
        if self.wandb_run is not None:
            stats = getattr(self, "_fixed_8plus2_last_stats", None)
            if stats is not None:
                self.wandb_run.log(
                    {
                        "replay/fixed_recent_views": float(stats["recent_views"]),
                        "replay/fixed_history_views": float(stats["historical_views"]),
                        "replay/historical_pool_size": float(stats["historical_pool_size"]),
                        "replay/fixed_total_active_views": float(stats["total_active_views"]),
                    },
                    step=self.global_iteration,
                )

    optimize_fixed_8plus2._fixed_8plus2_patch = True  # type: ignore[attr-defined]
    record_evaluation_fixed_8plus2._fixed_8plus2_patch = True  # type: ignore[attr-defined]
    StreamingIncrementalBackend._optimize_active_map = optimize_fixed_8plus2
    StreamingIncrementalBackend._record_evaluation = record_evaluation_fixed_8plus2

    print(
        "[fixed 8+2] installed: 8 most recent + 2 randomly sampled historical views; "
        "standard optimizer/maintenance/pruning unchanged",
        flush=True,
    )


def main() -> None:
    _install_fixed_8plus2()
    import run_pipeline_execution_benchmark_repro as repro

    repro.main()


if __name__ == "__main__":
    main()

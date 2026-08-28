#!/usr/bin/env python3
"""Run the reproducible empty-safe replay benchmark with fixed recent-10 evaluation.

Optimization behavior is unchanged.  The wrapper only appends one final metric,
``fixed_recent_10``, evaluated on the last ten inserted training cameras
regardless of ``backend.local_map_size``.  This makes recent-view quality
comparable across local-window-size ablations.
"""
from __future__ import annotations

import run_pipeline_execution_benchmark_repro_empty_safe as empty_safe


def _install_fixed_recent10_final_eval() -> None:
    from async_pipeline.backend_evaluation import BackendEvaluationMixin

    original_finalize_impl = BackendEvaluationMixin._finalize_impl
    if getattr(original_finalize_impl, "_fixed_recent10_eval_patch", False):
        return

    def finalize_with_fixed_recent10(self):
        summary = original_finalize_impl(self)
        if self.config.evaluation_enabled and self.train_packet_count > 0:
            final_metrics = summary.setdefault("final_metrics", {})
            final_metrics["fixed_recent_10"] = self._evaluate(self.train_cameras[-10:])
            # The original finalize has already written this artifact; overwrite
            # it with the additional evaluation-only field.
            self._save_json("incremental_backend_summary.json", summary)
        return summary

    finalize_with_fixed_recent10._fixed_recent10_eval_patch = True  # type: ignore[attr-defined]
    BackendEvaluationMixin._finalize_impl = finalize_with_fixed_recent10


_install_fixed_recent10_final_eval()


if __name__ == "__main__":
    # Importing empty_safe has already replaced the replay installer in repro.
    empty_safe.repro.main()

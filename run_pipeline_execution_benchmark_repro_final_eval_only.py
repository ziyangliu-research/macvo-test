#!/usr/bin/env python3
"""Run the reproducible empty-safe replay benchmark with stream-time evaluation disabled.

Final Train/Test evaluation is preserved in backend.finalize().  Intermediate
metric rendering during packet processing is suppressed so frame_timing_log.json
measures only the actual online pipeline work.  FPS can therefore be computed
from the final backend_end_sec without including final metric rendering.
"""
from __future__ import annotations

import run_pipeline_execution_benchmark_repro_empty_safe as safe


def _disable_intermediate_evaluation() -> None:
    from async_pipeline.backend_evaluation import BackendEvaluationMixin

    original = BackendEvaluationMixin._record_evaluation
    if getattr(original, "_final_eval_only_patch", False):
        return

    def no_intermediate_evaluation(self, stage, update, active_cameras):
        return None

    no_intermediate_evaluation._final_eval_only_patch = True  # type: ignore[attr-defined]
    BackendEvaluationMixin._record_evaluation = no_intermediate_evaluation
    print(
        "[repro] intermediate evaluation rendering disabled; final Train/Test evaluation preserved",
        flush=True,
    )


if __name__ == "__main__":
    _disable_intermediate_evaluation()
    safe.repro.main()

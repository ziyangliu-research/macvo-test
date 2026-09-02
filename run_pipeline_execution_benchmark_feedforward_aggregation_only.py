#!/usr/bin/env python3
"""Feed-forward aggregation-only benchmark.

Keeps the exact normal serial pipeline through:
  stereo input -> MAC-VO pose -> ReSplat packet -> world transform -> append.

Disables GraphDECO optimization and maintenance entirely. Intermediate metric
rendering is also disabled, while the normal final Train/Test evaluation in
backend.finalize() is preserved.

For a true feed-forward baseline, launch with
backend.reset_new_packet_opacity=false so ReSplat-predicted opacities are kept.
"""
from __future__ import annotations

import run_pipeline_execution_benchmark_repro_empty_safe as safe


def _install_feedforward_only() -> None:
    from async_pipeline.backend_core import StreamingIncrementalBackend
    from async_pipeline.backend_evaluation import BackendEvaluationMixin

    def no_optimization(self, update, active_cameras):
        return None

    no_optimization._feedforward_aggregation_only_patch = True  # type: ignore[attr-defined]
    StreamingIncrementalBackend._optimize_active_map = no_optimization

    def no_intermediate_evaluation(self, stage, update, active_cameras):
        return None

    no_intermediate_evaluation._final_eval_only_patch = True  # type: ignore[attr-defined]
    BackendEvaluationMixin._record_evaluation = no_intermediate_evaluation

    print(
        "[feedforward-only] GraphDECO optimization/maintenance disabled; "
        "final Train/Test evaluation preserved",
        flush=True,
    )


if __name__ == "__main__":
    _install_feedforward_only()
    safe.repro.main()

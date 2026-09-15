#!/usr/bin/env python3
"""Reproducible component-ablation runner for SH003.

Supported modes via PIPELINE_COMPONENT_ABLATION_MODE:
  ff_only       : append feed-forward packets only; no optimizer updates, no maintenance
  incremental   : recent-view incremental optimization + standard maintenance
  strong_prune  : same as incremental; launcher sets a stronger opacity threshold
  historical    : incremental optimization with post-maintenance historical replay
  full          : strong pruning + post-maintenance historical replay

This wrapper preserves the formal serial/default-stream ReSplat path, disables
intermediate metric rendering, and uses zero-Gaussian-safe optimization for both
recent-only and replay variants. Final Train/Test metrics are still evaluated.
"""
from __future__ import annotations

import os

import run_pipeline_execution_benchmark_repro_empty_safe as safe
from run_pipeline_execution_benchmark_repro_test_psnr_only import (
    _install_no_replay_empty_map_guard,
)


def _disable_intermediate_evaluation() -> None:
    from async_pipeline.backend_evaluation import BackendEvaluationMixin

    original = BackendEvaluationMixin._record_evaluation
    if getattr(original, "_component_ablation_final_eval_only", False):
        return

    def no_intermediate_evaluation(self, stage, update, active_cameras):
        return None

    no_intermediate_evaluation._component_ablation_final_eval_only = True  # type: ignore[attr-defined]
    BackendEvaluationMixin._record_evaluation = no_intermediate_evaluation
    print(
        "[component ablation] intermediate evaluation disabled; final Train/Test evaluation preserved",
        flush=True,
    )


def _install_ff_only() -> None:
    """Replace backend optimization with a no-op while preserving packet insertion."""
    from async_pipeline.backend_core import StreamingIncrementalBackend

    original = StreamingIncrementalBackend._optimize_active_map
    if getattr(original, "_ff_only_component_patch", False):
        return

    def ff_only_no_optimization(self, update, active_cameras):
        # Intentionally do not advance global_iteration: no optimizer updates occur.
        # Packet conversion/alignment and append remain unchanged.
        return None

    ff_only_no_optimization._ff_only_component_patch = True  # type: ignore[attr-defined]
    StreamingIncrementalBackend._optimize_active_map = ff_only_no_optimization
    print(
        "[component ablation] FF-only mode: packet insertion/alignment only; no optimization or maintenance",
        flush=True,
    )


def main() -> None:
    mode = os.environ.get("PIPELINE_COMPONENT_ABLATION_MODE", "").strip().lower()
    valid = {"ff_only", "incremental", "strong_prune", "historical", "full"}
    if mode not in valid:
        raise ValueError(
            "PIPELINE_COMPONENT_ABLATION_MODE must be one of " + ", ".join(sorted(valid))
        )

    _disable_intermediate_evaluation()

    replay_raw = os.environ.get("PIPELINE_HISTORICAL_REPLAY_FRACTION")
    replay_fraction = float(replay_raw) if replay_raw is not None else 0.0

    if mode == "ff_only":
        if replay_fraction > 0:
            raise ValueError("ff_only must not enable historical replay")
        _install_ff_only()
    elif replay_fraction <= 0.0:
        # Recent-only strong-pruning runs can temporarily prune to zero. Preserve
        # that result and skip only the invalid zero-Gaussian backward passes.
        _install_no_replay_empty_map_guard()

    safe.repro.main()


if __name__ == "__main__":
    main()

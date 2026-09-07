#!/usr/bin/env python3
"""Endpoint-only quality runner for the stride-downsampled EuRoC protocol."""
from __future__ import annotations

import run_async_pipeline_metrics as pose_metrics
from euroc_stride5_support import (
    evaluate_pose_euroc,
    install_euroc_runtime_support,
)

# Install dataset/runtime semantics before the normal benchmark is constructed.
install_euroc_runtime_support()
pose_metrics.evaluate_pose = evaluate_pose_euroc

import run_pipeline_execution_benchmark_repro_posthoc_global_refine_endpoint_lpips as quality


if __name__ == "__main__":
    quality._disable_intermediate_online_evaluation()
    quality._install_lpips_evaluator()
    quality._install_endpoint_refinement()
    quality.safe.repro.main()

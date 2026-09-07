#!/usr/bin/env python3
"""Timing-only runner for the stride-downsampled EuRoC protocol."""
from __future__ import annotations

from euroc_stride5_support import install_euroc_runtime_support

install_euroc_runtime_support()

import run_pipeline_execution_benchmark_repro_posthoc_global_refine_timing_only as timing


if __name__ == "__main__":
    timing._disable_all_evaluation()
    timing._install_timing_refinement()
    timing.safe.repro.main()

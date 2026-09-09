#!/usr/bin/env python3
"""TartanAir wrapper for native-GraphDECO post-hoc refinement + online test renders."""
from __future__ import annotations

from online_test_render_export import install_online_test_render_export
import run_pipeline_execution_benchmark_repro_posthoc_global_refine_native3dgs_lpips as native


if __name__ == "__main__":
    native.quality_helpers._disable_intermediate_online_evaluation()
    native.quality_helpers._install_lpips_evaluator()
    install_online_test_render_export()
    native._install_native_graphdeco_refinement()
    native.safe.repro.main()

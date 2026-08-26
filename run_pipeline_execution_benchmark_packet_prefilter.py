#!/usr/bin/env python3
"""Run the reproducible pipeline with local ReSplat packet pre-optimization/pruning.

The normal MAC-VO + ReSplat + incremental GraphDECO baseline remains unchanged
except for one operation inserted immediately after ReSplat inference and before
packet handoff to the global backend:

    local packet -> opacity cap -> short stereo 3DGS optimization -> opacity prune

The filtered packet is still processed by the global backend exactly like a
normal new packet, including the baseline global opacity reset and historical
replay policy.

Environment variables:

PIPELINE_PACKET_PREFILTER_ITERS
    Local stereo optimization iterations. Default: 10.
PIPELINE_PACKET_PREFILTER_POST_PRUNE_ITERS
    Optional extra local iterations after pruning. Default: 0.
PIPELINE_PACKET_PREFILTER_RESET_MAX_OPACITY
    Local prefilter opacity cap. Default: 0.01.
PIPELINE_PACKET_PREFILTER_PRUNE_MIN_OPACITY
    Local packet opacity prune threshold. Default: 0.005.
PIPELINE_PACKET_PREFILTER_LOG_EVERY_ITERATION
    1/true to print every local optimization iteration. Default: true.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import run_async_pipeline as base
import run_pipeline_execution_benchmark_repro as repro

from async_pipeline.local_packet_prefilter import (
    LocalPacketPrefilter,
    LocalPacketPrefilterConfig,
)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _peek_resolved_config() -> dict[str, Any]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    known, _ = parser.parse_known_args()
    if not known.config:
        raise RuntimeError("--config is required for packet-prefilter benchmark")

    root = Path(__file__).resolve().parent
    config = base.load_yaml(base.absolute(known.config, root))
    for item in known.overrides:
        if "=" not in item:
            raise ValueError(f"invalid override {item!r}")
        key, raw = item.split("=", 1)
        base.nested_set(config, key, base.normalize_override_value(key, raw))
    return base.resolve(config, root)


def _install_packet_prefilter(resolved: dict[str, Any]) -> None:
    from async_pipeline.resplat_runtime import ResplatPacketGenerator

    original_infer = ResplatPacketGenerator.infer
    if getattr(original_infer, "_local_packet_prefilter_patch", False):
        return

    backend = resolved["backend"]
    work_dir = Path(resolved["paths"]["work_dir"])
    config = LocalPacketPrefilterConfig(
        gs_repo=Path(resolved["paths"]["gs_repo"]),
        work_dir=work_dir,
        device=str(resolved["resplat_frontend"]["device"]),
        sh_degree=int(backend["sh_degree"]),
        iterations=int(os.environ.get("PIPELINE_PACKET_PREFILTER_ITERS", "10")),
        post_prune_iterations=int(
            os.environ.get("PIPELINE_PACKET_PREFILTER_POST_PRUNE_ITERS", "0")
        ),
        reset_max_opacity=float(
            os.environ.get("PIPELINE_PACKET_PREFILTER_RESET_MAX_OPACITY", "0.01")
        ),
        prune_min_opacity=float(
            os.environ.get("PIPELINE_PACKET_PREFILTER_PRUNE_MIN_OPACITY", "0.005")
        ),
        spatial_lr_scale=float(backend["spatial_lr_scale"]),
        white_background=bool(backend["white_background"]),
        antialiasing=bool(backend["antialiasing"]),
        log_every_iteration=_env_bool(
            "PIPELINE_PACKET_PREFILTER_LOG_EVERY_ITERATION", True
        ),
    )
    optimization = dict(backend["optimization"])
    prefilter: LocalPacketPrefilter | None = None

    def infer_with_packet_prefilter(self: ResplatPacketGenerator, *args, **kwargs):
        nonlocal prefilter
        result = original_infer(self, *args, **kwargs)
        if prefilter is None:
            prefilter = LocalPacketPrefilter(config, optimization)
            print(
                "[packet prefilter] enabled: "
                f"iters={config.iterations}, "
                f"post_prune_iters={config.post_prune_iterations}, "
                f"reset={config.reset_max_opacity}, "
                f"prune={config.prune_min_opacity}, "
                "supervision=alternating stereo, densification=off, "
                "global backend unchanged",
                flush=True,
            )
        return prefilter.process(result)

    infer_with_packet_prefilter._local_packet_prefilter_patch = True  # type: ignore[attr-defined]
    ResplatPacketGenerator.infer = infer_with_packet_prefilter


if __name__ == "__main__":
    resolved = _peek_resolved_config()
    _install_packet_prefilter(resolved)
    repro.main()

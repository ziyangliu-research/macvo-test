#!/usr/bin/env python3
"""Seed benchmark RNGs and run the reproducible execution baseline.

For the formal serial baseline, ReSplat is pinned to CUDA's default stream.
The current ReSplat stack has operators that do not behave correctly when the
whole encoder is launched on a separate persistent CUDA stream: raw packet
self-render quality collapses despite explicit inter-stream waits. The validated
legacy ZipMap+ReSplat path and the default-stream diagnostic both produce the
expected packet quality.

Async scheduling remains available for future investigation, but this wrapper
only applies the stream override when ``--mode serial`` is requested.

Optional root-cause diagnostics can be enabled with
``PIPELINE_ROOT_CAUSE_DIAGNOSTICS=1``. They do not change the optimization
algorithm. At selected train-packet ordinals they render frozen historical
camera sets at five stages:

- S0_pre_append: before the new packet is appended;
- S1_post_append: immediately after append, before optimization;
- S2_pre_maintenance: after the optimizer step at the configured maintenance
  iteration, immediately before densify/prune;
- S3_post_maintenance: immediately after densify/prune;
- S4_post_optimization: after all per-packet optimization iterations.

The same already-seen cameras are used from S0 through S4 so that stage deltas
measure changes to previously reconstructed views rather than changes in the
camera evaluation set. Results are written to ``diagnostic_stage_log.json`` in
the incremental backend output directory.
"""
from __future__ import annotations

import os
import random
import sys
from typing import Any

import numpy as np
import torch


def _requested_mode(argv: list[str]) -> str | None:
    for index, token in enumerate(argv):
        if token == "--mode" and index + 1 < len(argv):
            return argv[index + 1]
        if token.startswith("--mode="):
            return token.split("=", 1)[1]
    return None


def _pin_serial_resplat_to_default_stream() -> None:
    """Make the formal serial baseline use the validated ReSplat stream path."""

    from async_pipeline.resplat_runtime import ResplatPacketGenerator

    original_initialize = ResplatPacketGenerator.initialize
    if getattr(original_initialize, "_serial_default_stream_patch", False):
        return

    def initialize_on_default_stream(self: ResplatPacketGenerator) -> None:
        original_initialize(self)
        if self.device.type == "cuda":
            self.stream = torch.cuda.default_stream(self.device)

    initialize_on_default_stream._serial_default_stream_patch = True  # type: ignore[attr-defined]
    ResplatPacketGenerator.initialize = initialize_on_default_stream


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _parse_positive_int_list(raw: str) -> tuple[int, ...]:
    values: list[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value <= 0:
            raise ValueError(f"diagnostic packet ordinal must be positive, got {value}")
        values.append(value)
    if not values:
        raise ValueError("PIPELINE_DIAGNOSTIC_PACKETS must contain at least one integer")
    return tuple(sorted(set(values)))


def _install_root_cause_diagnostics() -> None:
    """Instrument the incremental backend without changing its training policy.

    Diagnostics are intentionally installed here rather than in backend_core.py
    so that the normal pipeline remains untouched unless the reproducibility
    wrapper is explicitly launched with PIPELINE_ROOT_CAUSE_DIAGNOSTICS=1.
    """

    from async_pipeline.backend_core import StreamingIncrementalBackend

    original_append = StreamingIncrementalBackend._append_gaussians
    if getattr(original_append, "_root_cause_diagnostic_patch", False):
        return

    original_optimize = StreamingIncrementalBackend._optimize_active_map
    original_maintenance = StreamingIncrementalBackend._run_maintenance

    packet_ordinals = _parse_positive_int_list(
        os.environ.get("PIPELINE_DIAGNOSTIC_PACKETS", "10,20,30,40")
    )
    early_view_count = int(os.environ.get("PIPELINE_DIAGNOSTIC_EARLY_VIEWS", "10"))
    if early_view_count <= 0:
        raise ValueError("PIPELINE_DIAGNOSTIC_EARLY_VIEWS must be positive")

    def _diagnostic_log(self: StreamingIncrementalBackend) -> list[dict[str, Any]]:
        log = getattr(self, "_root_cause_diagnostic_log", None)
        if log is None:
            log = []
            setattr(self, "_root_cause_diagnostic_log", log)
        return log

    def _current_xyz_lr(self: StreamingIncrementalBackend) -> float | None:
        if self.gaussians is None or getattr(self.gaussians, "optimizer", None) is None:
            return None
        for group in self.gaussians.optimizer.param_groups:
            if group.get("name") == "xyz":
                return float(group.get("lr", 0.0))
        return None

    def _record_stage(
        self: StreamingIncrementalBackend,
        stage: str,
        stage_order: int,
    ) -> None:
        context = getattr(self, "_root_cause_diagnostic_context", None)
        if not context:
            return

        frozen_history = context["frozen_history_cameras"]
        fixed_early = frozen_history[:early_view_count]
        recent_history = frozen_history[-self.config.local_map_size :]

        metrics: dict[str, Any] = {
            "fixed_early_history": self._evaluate(fixed_early),
            "recent_history": self._evaluate(recent_history),
            "all_history": self._evaluate(frozen_history),
        }

        # The exact camera set used by the optimizer becomes available only after
        # the new camera is inserted into train_cameras. It is therefore logged
        # for S2-S4 as supplementary information, while the three frozen-history
        # sets above remain directly comparable across all five stages.
        optimization_active = context.get("optimization_active_cameras")
        if optimization_active is not None and stage_order >= 2:
            metrics["optimization_active"] = self._evaluate(optimization_active)

        entry: dict[str, Any] = {
            "diagnostic_packet_ordinal": int(context["packet_ordinal"]),
            "frame_index": context.get("frame_index"),
            "stage": stage,
            "stage_order": int(stage_order),
            "global_iteration": int(self.global_iteration),
            "num_gaussians": int(self.gaussians.get_xyz.shape[0]),
            "xyz_lr": _current_xyz_lr(self),
            "frozen_history_num_views": len(frozen_history),
            "fixed_early_requested_views": early_view_count,
            "local_map_size": int(self.config.local_map_size),
            "metrics": metrics,
        }
        log = _diagnostic_log(self)
        log.append(entry)
        context.setdefault("log_indices", []).append(len(log) - 1)

        if self.config.write_runtime_artifacts:
            self._save_json("diagnostic_stage_log.json", log)

        early = metrics["fixed_early_history"]
        recent = metrics["recent_history"]
        all_history = metrics["all_history"]
        print(
            "[root-cause diagnostic] "
            f"packet={entry['diagnostic_packet_ordinal']} "
            f"frame={entry['frame_index']} "
            f"stage={stage} "
            f"iter={entry['global_iteration']} "
            f"G={entry['num_gaussians']} "
            f"xyz_lr={entry['xyz_lr']} "
            f"early_psnr={early.get('psnr')} "
            f"recent_history_psnr={recent.get('psnr')} "
            f"all_history_psnr={all_history.get('psnr')}",
            flush=True,
        )

    def append_with_diagnostics(
        self: StreamingIncrementalBackend,
        tensors: dict[str, torch.Tensor],
    ) -> None:
        packet_ordinal = int(self.train_packet_count) + 1
        enabled = packet_ordinal in packet_ordinals
        if enabled:
            # Freeze only cameras that were already part of the map before this
            # packet arrived. This makes S0->S1 a pure append/interference test.
            context = {
                "packet_ordinal": packet_ordinal,
                "frame_index": None,
                "frozen_history_cameras": list(self.train_cameras),
                "optimization_active_cameras": None,
                "log_indices": [],
            }
            setattr(self, "_root_cause_diagnostic_context", context)
            _record_stage(self, "S0_pre_append", 0)

        result = original_append(self, tensors)

        if enabled:
            _record_stage(self, "S1_post_append", 1)
        return result

    append_with_diagnostics._root_cause_diagnostic_patch = True  # type: ignore[attr-defined]

    def maintenance_with_diagnostics(
        self: StreamingIncrementalBackend,
        update,
        local_iteration: int,
        radii: torch.Tensor,
    ):
        context = getattr(self, "_root_cause_diagnostic_context", None)
        if context:
            _record_stage(self, "S2_pre_maintenance", 2)

        event = original_maintenance(self, update, local_iteration, radii)

        if context:
            _record_stage(self, "S3_post_maintenance", 3)
        return event

    def optimize_with_diagnostics(
        self: StreamingIncrementalBackend,
        update,
        active_cameras,
    ):
        context = getattr(self, "_root_cause_diagnostic_context", None)
        if context:
            context["frame_index"] = int(update.descriptor.frame_index)
            context["optimization_active_cameras"] = list(active_cameras)
            # Backfill S0/S1 with the frame index, which is not visible inside
            # _append_gaussians itself.
            log = _diagnostic_log(self)
            for index in context.get("log_indices", []):
                log[index]["frame_index"] = int(update.descriptor.frame_index)
            if self.config.write_runtime_artifacts:
                self._save_json("diagnostic_stage_log.json", log)

        event = original_optimize(self, update, active_cameras)

        if context:
            _record_stage(self, "S4_post_optimization", 4)
            setattr(self, "_root_cause_diagnostic_context", None)
        return event

    StreamingIncrementalBackend._append_gaussians = append_with_diagnostics
    StreamingIncrementalBackend._run_maintenance = maintenance_with_diagnostics
    StreamingIncrementalBackend._optimize_active_map = optimize_with_diagnostics

    print(
        "[repro] root-cause diagnostics enabled: "
        f"packets={packet_ordinals}, early_views={early_view_count}",
        flush=True,
    )


def main() -> None:
    seed = int(os.environ.get("PIPELINE_BENCHMARK_SEED", "0"))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))

    mode = _requested_mode(sys.argv[1:])
    if mode == "serial":
        _pin_serial_resplat_to_default_stream()
        print(
            "[repro] serial baseline: ReSplat pinned to CUDA default stream",
            flush=True,
        )

    if _env_flag("PIPELINE_ROOT_CAUSE_DIAGNOSTICS"):
        _install_root_cause_diagnostics()

    from run_pipeline_execution_benchmark import main as benchmark_main

    benchmark_main()


if __name__ == "__main__":
    main()

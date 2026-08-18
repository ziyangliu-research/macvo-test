#!/usr/bin/env python3
"""Run the serial execution benchmark while replacing emitted MAC-VO poses with GT.

This is a diagnostic control for isolating pose/alignment errors from ReSplat and
GraphDECO backend behavior. The normal frame loader, ReSplat packet generation,
packet conversion, maintenance, optimization, and evaluation paths are unchanged.
Only the committed ``T_world_from_left`` supplied to the joiner/backend is replaced
by ground-truth TartanAir poses.

The wrapper delegates to ``run_pipeline_execution_benchmark_repro.py`` so the
validated serial ReSplat default-stream behavior is preserved.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


def _pop_option(argv: list[str], name: str) -> str | None:
    """Remove one wrapper-only CLI option before delegating to the baseline parser."""
    prefix = name + "="
    for i, token in enumerate(list(argv)):
        if token.startswith(prefix):
            value = token[len(prefix) :]
            del argv[i]
            return value
        if token == name:
            if i + 1 >= len(argv):
                raise ValueError(f"{name} requires a value")
            value = argv[i + 1]
            del argv[i : i + 2]
            return value
    return None


def _patch_gt_pose_emission(gt_pose_file: Path) -> None:
    from async_pipeline.contracts import PoseEstimate
    from async_pipeline.macvo_runtime import (
        MacvoPoseFrontend,
        _pose7_xyzw_to_matrix,
        _tartan_from_cv,
    )

    rows = np.loadtxt(gt_pose_file, dtype=np.float64)
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)
    if rows.ndim != 2 or rows.shape[1] < 7:
        raise ValueError(
            f"expected GT rows [tx ty tz qx qy qz qw], got shape {rows.shape}"
        )
    rows = rows[:, :7]

    original_emit_until = MacvoPoseFrontend._emit_until
    if getattr(original_emit_until, "_gt_pose_diagnostic_patch", False):
        return

    def gt_relative_pose(self: MacvoPoseFrontend, local_index: int) -> torch.Tensor:
        original_index = int(self.config.start_index) + int(local_index)
        first_index = int(self.config.start_index)
        if first_index < 0 or original_index >= rows.shape[0]:
            raise IndexError(
                f"GT pose index {original_index} outside {gt_pose_file} "
                f"with {rows.shape[0]} rows"
            )
        tartan_from_cv = _tartan_from_cv(dtype=torch.float64)
        first_abs = (
            _pose7_xyzw_to_matrix(torch.from_numpy(rows[first_index]))
            @ tartan_from_cv
        )
        current_abs = (
            _pose7_xyzw_to_matrix(torch.from_numpy(rows[original_index]))
            @ tartan_from_cv
        )
        return (torch.linalg.inv(first_abs) @ current_abs).float()

    def emit_gt_until(self: MacvoPoseFrontend, count: int) -> list[PoseEstimate]:
        output: list[PoseEstimate] = []
        while self._next_emit < count:
            index = self._next_emit
            descriptor = self._descriptors[index]
            estimate = PoseEstimate(
                descriptor=descriptor,
                T_world_from_left=gt_relative_pose(self, index),
                valid=True,
                committed=True,
                revision=0,
                latency_sec=self._last_run_sec,
                metadata={
                    "source": "TartanAir ground truth diagnostic override",
                    "coordinate_convention": "metric OpenCV c2w, first selected frame identity",
                    "gt_pose_file": str(gt_pose_file),
                    "replaces_macvo_pose": True,
                },
            )
            estimate.validate()
            output.append(estimate)
            self._next_emit += 1
        return output

    emit_gt_until._gt_pose_diagnostic_patch = True  # type: ignore[attr-defined]
    MacvoPoseFrontend._emit_until = emit_gt_until


def main() -> None:
    gt_value = _pop_option(sys.argv, "--gt_pose_file")
    if gt_value is None:
        raise SystemExit(
            "This diagnostic requires --gt_pose_file /absolute/path/to/SH003.txt"
        )
    gt_pose_file = Path(gt_value).expanduser().resolve()
    if not gt_pose_file.is_file():
        raise FileNotFoundError(f"GT pose file not found: {gt_pose_file}")

    _patch_gt_pose_emission(gt_pose_file)
    print(
        f"[gt-pose diagnostic] replacing committed MAC-VO poses with {gt_pose_file}",
        flush=True,
    )

    # This wrapper also preserves the serial ReSplat default-stream fix.
    from run_pipeline_execution_benchmark_repro import main as baseline_main

    baseline_main()


if __name__ == "__main__":
    main()

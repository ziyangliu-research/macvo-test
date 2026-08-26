#!/usr/bin/env python3
"""Stereo packet recovery curve with read-only multi-threshold prune diagnostics.

This wrapper keeps run_pipeline_packet_recovery_curve.py's optimization exactly
unchanged.  It only augments each recovery checkpoint with hypothetical opacity
prune counts for several thresholds.  No pruning is performed during the curve,
so all checkpoints describe one uninterrupted optimization trajectory.

The primary threshold used by the parent diagnostic is still controlled by
PIPELINE_PACKET_RECOVERY_PRUNE_THRESHOLD (use 0.1 for the current experiment).
Additional thresholds are controlled by:

PIPELINE_PACKET_RECOVERY_PRUNE_THRESHOLDS
    Comma-separated list. Default: 0.005,0.01,0.02,0.03,0.05,0.1
"""
from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any

import torch

import run_pipeline_packet_recovery_curve as base_curve


def _parse_thresholds(raw: str) -> tuple[float, ...]:
    values = sorted({float(x.strip()) for x in raw.split(",") if x.strip()})
    if not values:
        raise ValueError("PIPELINE_PACKET_RECOVERY_PRUNE_THRESHOLDS must not be empty")
    for value in values:
        if not 0.0 <= value < 1.0:
            raise ValueError(f"invalid prune threshold {value}; expected [0,1)")
    return tuple(values)


class MultiThresholdPacketRecoveryCurve(base_curve.PacketRecoveryCurve):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.prune_thresholds = _parse_thresholds(
            os.environ.get(
                "PIPELINE_PACKET_RECOVERY_PRUNE_THRESHOLDS",
                "0.005,0.01,0.02,0.03,0.05,0.1",
            )
        )

    @torch.no_grad()
    def _checkpoint(self, g: Any, cameras: list[Any], iteration: int) -> dict[str, Any]:
        row = super()._checkpoint(g, cameras, iteration)
        opacity = g.get_opacity.detach().reshape(-1)
        count = int(opacity.numel())
        sweep: dict[str, dict[str, float | int]] = {}
        for threshold in self.prune_thresholds:
            would_prune = int((opacity < threshold).sum().item()) if count else 0
            key = f"{threshold:.6g}"
            sweep[key] = {
                "threshold": float(threshold),
                "would_prune_count": would_prune,
                "would_prune_ratio": 0.0 if count == 0 else float(would_prune / count),
                "would_keep_count": count - would_prune,
            }
        row["hypothetical_prune_multi"] = sweep
        return row

    def _write_recovery(self, payload: dict[str, Any]) -> None:
        # Preserve the parent's JSON and primary-threshold CSV.
        super()._write_recovery(payload)

        output = self.config.work_dir / "packet_recovery_curve_multithreshold.csv"
        threshold_keys = [f"{value:.6g}" for value in self.prune_thresholds]
        columns = [
            "iteration",
            "left_psnr",
            "right_psnr",
            "mean_psnr",
            "mean_ssim",
            "opacity_mean",
            "opacity_p10",
            "opacity_p50",
            "opacity_p90",
        ]
        for key in threshold_keys:
            safe = key.replace(".", "p")
            columns.extend(
                [
                    f"th_{safe}_would_prune_count",
                    f"th_{safe}_would_prune_ratio",
                    f"th_{safe}_would_keep_count",
                ]
            )

        with output.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            for row in payload["checkpoints"]:
                stereo = row["stereo_metrics"]
                opacity = row["opacity"]
                out: dict[str, Any] = {
                    "iteration": row["iteration"],
                    "left_psnr": stereo["left"]["psnr"],
                    "right_psnr": stereo["right"]["psnr"],
                    "mean_psnr": stereo["mean"]["psnr"],
                    "mean_ssim": stereo["mean"]["ssim"],
                    "opacity_mean": opacity["mean"],
                    "opacity_p10": opacity["quantiles"]["p10"],
                    "opacity_p50": opacity["quantiles"]["p50"],
                    "opacity_p90": opacity["quantiles"]["p90"],
                }
                sweep = row["hypothetical_prune_multi"]
                for key in threshold_keys:
                    safe = key.replace(".", "p")
                    stats = sweep[key]
                    out[f"th_{safe}_would_prune_count"] = stats["would_prune_count"]
                    out[f"th_{safe}_would_prune_ratio"] = stats["would_prune_ratio"]
                    out[f"th_{safe}_would_keep_count"] = stats["would_keep_count"]
                writer.writerow(out)

        payload["hypothetical_prune_thresholds"] = list(self.prune_thresholds)
        # Re-write JSON after adding the threshold list for self-description.
        self.recovery_json_path.write_text(
            __import__("json").dumps(payload, indent=2), encoding="utf-8"
        )
        print(
            "[packet recovery multithreshold] "
            f"thresholds={self.prune_thresholds} csv={output}",
            flush=True,
        )


# _install_recovery resolves PacketRecoveryCurve from the module global at call time.
base_curve.PacketRecoveryCurve = MultiThresholdPacketRecoveryCurve


if __name__ == "__main__":
    resolved = base_curve._peek_resolved_config()
    base_curve._install_recovery(resolved)
    base_curve.repro.main()

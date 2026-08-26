#!/usr/bin/env python3
"""Diagnose recovery of one ReSplat packet after opacity reset.

This experiment does NOT prune or modify the packet handed to the downstream
pipeline. It creates a temporary GraphDECO GaussianModel from one ReSplat local
packet, caps opacity, optimizes continuously, and records recovery checkpoints.

Default checkpoints: 0,10,20,50,100,150,200.
Checkpoint 0 is immediately after opacity reset and before any optimizer step.
The raw pre-reset packet is recorded separately.

For direct comparison with the incremental backend's first packet, the default
supervision mode is ``left``: all optimization steps use the left camera only.
``stereo`` alternates left/right views.

Environment variables:
PIPELINE_PACKET_RECOVERY_FRAME          target frame index, default 0
PIPELINE_PACKET_RECOVERY_MAX_ITERS      max optimizer steps, default 200
PIPELINE_PACKET_RECOVERY_CHECKPOINTS    comma list, default 0,10,20,50,100,150,200
PIPELINE_PACKET_RECOVERY_SUPERVISION    left or stereo, default left
PIPELINE_PACKET_RECOVERY_RESET_OPACITY  opacity cap, default 0.01
PIPELINE_PACKET_RECOVERY_PRUNE_THRESHOLD hypothetical prune threshold, default 0.005
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

import torch

import run_async_pipeline as base
import run_pipeline_execution_benchmark_repro as repro
import run_pipeline_execution_benchmark_packet_prefilter_v2 as prefilter_v2

LocalPacketPrefilter = prefilter_v2.impl.LocalPacketPrefilter
LocalPacketPrefilterConfig = prefilter_v2.impl.LocalPacketPrefilterConfig


def _parse_checkpoints(raw: str, max_iters: int) -> tuple[int, ...]:
    values = sorted({int(x.strip()) for x in raw.split(",") if x.strip()})
    if not values:
        raise ValueError("recovery checkpoints must not be empty")
    if values[0] < 0 or values[-1] > max_iters:
        raise ValueError(
            f"recovery checkpoints must lie in [0,{max_iters}], got {values}"
        )
    return tuple(values)


def _peek_resolved_config() -> dict[str, Any]:
    # Reuse the already validated parser from the packet-prefilter runner.
    return prefilter_v2.impl._peek_resolved_config()


class PacketRecoveryCurve(LocalPacketPrefilter):
    def __init__(
        self,
        config: LocalPacketPrefilterConfig,
        optimization: dict[str, Any],
        *,
        target_frame: int,
        max_iters: int,
        checkpoints: tuple[int, ...],
        supervision: str,
        prune_threshold: float,
    ) -> None:
        super().__init__(config, optimization)
        self.target_frame = int(target_frame)
        self.max_iters = int(max_iters)
        self.checkpoints = checkpoints
        self.supervision = supervision
        self.prune_threshold = float(prune_threshold)
        self.completed = False

    @property
    def recovery_json_path(self) -> Path:
        return self.config.work_dir / "packet_recovery_curve.json"

    @property
    def recovery_csv_path(self) -> Path:
        return self.config.work_dir / "packet_recovery_curve.csv"

    @torch.no_grad()
    def _checkpoint(self, g: Any, cameras: list[Any], iteration: int) -> dict[str, Any]:
        stereo = self._evaluate_stereo(g, cameras)
        opacity = g.get_opacity.detach().reshape(-1)
        count = int(opacity.numel())
        prune_mask = opacity < self.prune_threshold
        would_prune = int(prune_mask.sum().item())

        if count:
            q_values = torch.quantile(
                opacity.float(),
                torch.tensor(
                    [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99],
                    device=opacity.device,
                ),
            ).tolist()
        else:
            q_values = [0.0] * 9

        lrs = {
            str(group.get("name", f"group_{i}")): float(group.get("lr", 0.0))
            for i, group in enumerate(g.optimizer.param_groups)
        }

        return {
            "iteration": int(iteration),
            "num_gaussians": count,
            "stereo_metrics": stereo,
            "opacity": {
                "mean": float(opacity.mean().item()) if count else 0.0,
                "min": float(opacity.min().item()) if count else 0.0,
                "max": float(opacity.max().item()) if count else 0.0,
                "lt_0.005_count": int((opacity < 0.005).sum().item()) if count else 0,
                "lt_0.005_ratio": float((opacity < 0.005).float().mean().item()) if count else 0.0,
                "lt_0.01_count": int((opacity < 0.01).sum().item()) if count else 0,
                "lt_0.01_ratio": float((opacity < 0.01).float().mean().item()) if count else 0.0,
                "lt_0.02_count": int((opacity < 0.02).sum().item()) if count else 0,
                "lt_0.02_ratio": float((opacity < 0.02).float().mean().item()) if count else 0.0,
                "lt_0.05_count": int((opacity < 0.05).sum().item()) if count else 0,
                "lt_0.05_ratio": float((opacity < 0.05).float().mean().item()) if count else 0.0,
                "quantiles": {
                    "p01": float(q_values[0]),
                    "p05": float(q_values[1]),
                    "p10": float(q_values[2]),
                    "p25": float(q_values[3]),
                    "p50": float(q_values[4]),
                    "p75": float(q_values[5]),
                    "p90": float(q_values[6]),
                    "p95": float(q_values[7]),
                    "p99": float(q_values[8]),
                },
            },
            "hypothetical_prune": {
                "threshold": self.prune_threshold,
                "would_prune_count": would_prune,
                "would_prune_ratio": 0.0 if count == 0 else float(would_prune / count),
                "would_keep_count": count - would_prune,
            },
            "learning_rates": lrs,
        }

    def _write_recovery(self, payload: dict[str, Any]) -> None:
        self.recovery_json_path.write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
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
            "lt_0p005_ratio",
            "would_prune_count",
            "would_prune_ratio",
        ]
        with self.recovery_csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            for row in payload["checkpoints"]:
                stereo = row["stereo_metrics"]
                opacity = row["opacity"]
                writer.writerow(
                    {
                        "iteration": row["iteration"],
                        "left_psnr": stereo["left"]["psnr"],
                        "right_psnr": stereo["right"]["psnr"],
                        "mean_psnr": stereo["mean"]["psnr"],
                        "mean_ssim": stereo["mean"]["ssim"],
                        "opacity_mean": opacity["mean"],
                        "opacity_p10": opacity["quantiles"]["p10"],
                        "opacity_p50": opacity["quantiles"]["p50"],
                        "opacity_p90": opacity["quantiles"]["p90"],
                        "lt_0p005_ratio": opacity["lt_0.005_ratio"],
                        "would_prune_count": row["hypothetical_prune"]["would_prune_count"],
                        "would_prune_ratio": row["hypothetical_prune"]["would_prune_ratio"],
                    }
                )

    def process_recovery(self, result: Any) -> Any:
        packet = result.packet
        frame = int(packet.descriptor.frame_index)
        if frame != self.target_frame or self.completed:
            return result

        self.initialize()
        g = self._make_model(packet)
        cameras = self._make_cameras(result)

        raw_metrics = self._evaluate_stereo(g, cameras)
        raw_opacity = self._opacity_stats(g)
        reset_stats = self._reset_opacity(g)

        checkpoints: list[dict[str, Any]] = []
        loss_curve: list[dict[str, Any]] = []

        if 0 in self.checkpoints:
            checkpoints.append(self._checkpoint(g, cameras, 0))

        print(
            "[packet recovery start] "
            f"frame={frame} G={int(g.get_xyz.shape[0])} "
            f"supervision={self.supervision} max_iters={self.max_iters} "
            f"raw_PSNR(L/R/M)={raw_metrics['left']['psnr']:.3f}/"
            f"{raw_metrics['right']['psnr']:.3f}/"
            f"{raw_metrics['mean']['psnr']:.3f} "
            f"opacity={reset_stats['before_mean']:.5f}->{reset_stats['after_mean']:.5f}",
            flush=True,
        )

        if checkpoints:
            c0 = checkpoints[-1]
            print(
                "[packet recovery checkpoint] "
                f"iter=0 PSNR(L/R/M)="
                f"{c0['stereo_metrics']['left']['psnr']:.3f}/"
                f"{c0['stereo_metrics']['right']['psnr']:.3f}/"
                f"{c0['stereo_metrics']['mean']['psnr']:.3f} "
                f"SSIM(M)={c0['stereo_metrics']['mean']['ssim']:.4f} "
                f"opacity_mean={c0['opacity']['mean']:.5f} "
                f"<0.005={100*c0['opacity']['lt_0.005_ratio']:.2f}% "
                f"would_prune={c0['hypothetical_prune']['would_prune_count']}",
                flush=True,
            )

        for iteration in range(1, self.max_iters + 1):
            camera_index = 0 if self.supervision == "left" else (iteration - 1) % 2
            train = self._optimize_step(g, cameras[camera_index], iteration)

            with torch.no_grad():
                opacity = g.get_opacity.detach().reshape(-1)
                loss_curve.append(
                    {
                        "iteration": iteration,
                        "supervision_view": "left" if camera_index == 0 else "right",
                        "loss": train["loss"],
                        "l1": train["l1"],
                        "ssim": train["ssim"],
                        "opacity_mean": float(opacity.mean().item()),
                        "lt_0.005_ratio": float((opacity < 0.005).float().mean().item()),
                    }
                )

            if iteration in self.checkpoints:
                row = self._checkpoint(g, cameras, iteration)
                checkpoints.append(row)
                print(
                    "[packet recovery checkpoint] "
                    f"iter={iteration} PSNR(L/R/M)="
                    f"{row['stereo_metrics']['left']['psnr']:.3f}/"
                    f"{row['stereo_metrics']['right']['psnr']:.3f}/"
                    f"{row['stereo_metrics']['mean']['psnr']:.3f} "
                    f"SSIM(M)={row['stereo_metrics']['mean']['ssim']:.4f} "
                    f"opacity_mean={row['opacity']['mean']:.5f} "
                    f"p10/p50/p90="
                    f"{row['opacity']['quantiles']['p10']:.5f}/"
                    f"{row['opacity']['quantiles']['p50']:.5f}/"
                    f"{row['opacity']['quantiles']['p90']:.5f} "
                    f"<0.005={100*row['opacity']['lt_0.005_ratio']:.2f}% "
                    f"would_prune={row['hypothetical_prune']['would_prune_count']}",
                    flush=True,
                )

        torch.cuda.synchronize(self.device)
        payload = {
            "frame_index": frame,
            "supervision": self.supervision,
            "max_iterations": self.max_iters,
            "checkpoint_iterations": list(self.checkpoints),
            "reset_max_opacity": self.config.reset_max_opacity,
            "hypothetical_prune_threshold": self.prune_threshold,
            "raw_stereo_metrics": raw_metrics,
            "raw_opacity": raw_opacity,
            "reset": reset_stats,
            "checkpoints": checkpoints,
            "loss_curve": loss_curve,
            "note": (
                "No pruning is applied during this diagnostic. Hypothetical prune "
                "counts are read-only opacity-threshold counts at each checkpoint. "
                "The original ReSplat packet is handed downstream unchanged."
            ),
        }
        self._write_recovery(payload)
        self.completed = True
        print(
            f"[packet recovery done] json={self.recovery_json_path} "
            f"csv={self.recovery_csv_path}",
            flush=True,
        )

        # Diagnostic only: do not replace the packet sent downstream.
        return result


def _install_recovery(resolved: dict[str, Any]) -> None:
    from async_pipeline.resplat_runtime import ResplatPacketGenerator

    target_frame = int(os.environ.get("PIPELINE_PACKET_RECOVERY_FRAME", "0"))
    max_iters = int(os.environ.get("PIPELINE_PACKET_RECOVERY_MAX_ITERS", "200"))
    if max_iters <= 0:
        raise ValueError("PIPELINE_PACKET_RECOVERY_MAX_ITERS must be positive")
    checkpoints = _parse_checkpoints(
        os.environ.get(
            "PIPELINE_PACKET_RECOVERY_CHECKPOINTS", "0,10,20,50,100,150,200"
        ),
        max_iters,
    )
    supervision = os.environ.get("PIPELINE_PACKET_RECOVERY_SUPERVISION", "left").strip().lower()
    if supervision not in {"left", "stereo"}:
        raise ValueError("PIPELINE_PACKET_RECOVERY_SUPERVISION must be left or stereo")
    reset_opacity = float(
        os.environ.get("PIPELINE_PACKET_RECOVERY_RESET_OPACITY", "0.01")
    )
    prune_threshold = float(
        os.environ.get("PIPELINE_PACKET_RECOVERY_PRUNE_THRESHOLD", "0.005")
    )

    backend = resolved["backend"]
    config = LocalPacketPrefilterConfig(
        gs_repo=Path(resolved["paths"]["gs_repo"]),
        work_dir=Path(resolved["paths"]["work_dir"]),
        device=str(resolved["resplat_frontend"]["device"]),
        sh_degree=int(backend["sh_degree"]),
        iterations=max_iters,
        post_prune_iterations=0,
        reset_max_opacity=reset_opacity,
        prune_min_opacity=prune_threshold,
        spatial_lr_scale=float(backend["spatial_lr_scale"]),
        white_background=bool(backend["white_background"]),
        antialiasing=bool(backend["antialiasing"]),
        log_every_iteration=False,
    )
    recovery = PacketRecoveryCurve(
        config,
        dict(backend["optimization"]),
        target_frame=target_frame,
        max_iters=max_iters,
        checkpoints=checkpoints,
        supervision=supervision,
        prune_threshold=prune_threshold,
    )

    original_infer = ResplatPacketGenerator.infer

    def infer_with_recovery(self: ResplatPacketGenerator, *args, **kwargs):
        result = original_infer(self, *args, **kwargs)
        return recovery.process_recovery(result)

    infer_with_recovery._packet_recovery_curve_patch = True  # type: ignore[attr-defined]
    ResplatPacketGenerator.infer = infer_with_recovery

    print(
        "[packet recovery] enabled: "
        f"frame={target_frame}, max_iters={max_iters}, checkpoints={checkpoints}, "
        f"supervision={supervision}, reset={reset_opacity}, "
        f"hypothetical_prune={prune_threshold}; actual_prune=OFF",
        flush=True,
    )


if __name__ == "__main__":
    resolved = _peek_resolved_config()
    _install_recovery(resolved)
    repro.main()

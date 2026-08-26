#!/usr/bin/env python3
"""Aggressive local-packet pruning recovery sweep.

Purpose
-------
Test whether a heavily compressed ReSplat local packet can recover quality when
post-prune optimization is allowed to continue substantially longer than the
20-step pilot experiment.

The experiment uses exactly one ReSplat packet and never replaces the packet
handed to the downstream pipeline.  It performs:

    ReSplat packet
      -> opacity cap
      -> stereo pre-optimization (default 150 steps)
      -> clone one independent branch per threshold
      -> ACTUAL opacity pruning
      -> stereo post-prune optimization
      -> recovery checkpoints 0/10/20/50/100/150/200

No densification is performed in this script.  This isolates representation
capacity after aggressive pruning.  A later experiment can add limited
post-prune densification only if prune-only recovery saturates too low.

Environment variables
---------------------
PIPELINE_PACKET_AGGR_FRAME
    Target frame index. Default: 0.
PIPELINE_PACKET_AGGR_PRE_ITERS
    Stereo optimization steps before pruning. Default: 150.
PIPELINE_PACKET_AGGR_THRESHOLDS
    Actual opacity prune thresholds. Default: 0.05,0.06,0.07,0.08,0.1.
PIPELINE_PACKET_AGGR_POST_MAX_ITERS
    Maximum post-prune optimization steps. Default: 200.
PIPELINE_PACKET_AGGR_POST_CHECKPOINTS
    Post-prune recovery checkpoints. Default: 0,10,20,50,100,150,200.
PIPELINE_PACKET_AGGR_RESET_OPACITY
    Opacity cap before pre-optimization. Default: 0.01.
PIPELINE_PACKET_AGGR_SUPERVISION
    left or stereo. Default: stereo.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

import torch

import run_pipeline_execution_benchmark_repro as repro
import run_pipeline_execution_benchmark_packet_prefilter_v2 as prefilter_v2
import run_pipeline_packet_actual_prune_sweep as actual_sweep

LocalPacketPrefilterConfig = prefilter_v2.impl.LocalPacketPrefilterConfig


def _parse_ints(raw: str, *, min_value: int, max_value: int) -> tuple[int, ...]:
    values = tuple(sorted({int(x.strip()) for x in raw.split(",") if x.strip()}))
    if not values:
        raise ValueError("integer list must not be empty")
    if values[0] < min_value or values[-1] > max_value:
        raise ValueError(
            f"values must lie in [{min_value},{max_value}], got {values}"
        )
    return values


def _parse_thresholds(raw: str) -> tuple[float, ...]:
    values = tuple(sorted({float(x.strip()) for x in raw.split(",") if x.strip()}))
    if not values:
        raise ValueError("threshold list must not be empty")
    for value in values:
        if not 0.0 <= value < 1.0:
            raise ValueError(f"invalid opacity threshold {value}; expected [0,1)")
    return values


class AggressivePruneRecoverySweep(actual_sweep.ActualPacketPruneSweep):
    def __init__(
        self,
        config: LocalPacketPrefilterConfig,
        optimization: dict[str, Any],
        *,
        target_frame: int,
        pre_iters: int,
        thresholds: tuple[float, ...],
        post_max_iters: int,
        post_checkpoints: tuple[int, ...],
        supervision: str,
    ) -> None:
        # Parent helpers are reused for exact model/Adam cloning and safe direct
        # GraphDECO pruning.  Parent checkpoint/post settings are not used.
        super().__init__(
            config,
            optimization,
            target_frame=target_frame,
            max_iters=pre_iters,
            checkpoints=(pre_iters,),
            thresholds=thresholds,
            post_iters=0,
            supervision=supervision,
        )
        self.pre_iters = int(pre_iters)
        self.post_max_iters = int(post_max_iters)
        self.post_checkpoints = post_checkpoints

    @property
    def output_json(self) -> Path:
        return self.config.work_dir / "packet_aggressive_prune_recovery_sweep.json"

    @property
    def output_csv(self) -> Path:
        return self.config.work_dir / "packet_aggressive_prune_recovery_sweep.csv"

    @torch.no_grad()
    def _snapshot(
        self,
        g: Any,
        cameras: list[Any],
        *,
        threshold: float,
        post_iteration: int,
    ) -> dict[str, Any]:
        opacity = self._opacity_stats(g)
        stereo = self._evaluate_stereo(g, cameras)
        return {
            "threshold": float(threshold),
            "post_iteration": int(post_iteration),
            "num_gaussians": int(g.get_xyz.shape[0]),
            "stereo_metrics": stereo,
            "opacity": opacity,
        }

    def _run_branch(
        self,
        source: Any,
        cameras: list[Any],
        threshold: float,
        pre_metrics: dict[str, Any],
    ) -> dict[str, Any]:
        branch = self._clone_model_with_optimizer(source)
        prune = self._prune_threshold(branch, threshold)

        result: dict[str, Any] = {
            "threshold": float(threshold),
            "pre_iterations": self.pre_iters,
            "pre_prune_stereo_metrics": pre_metrics,
            "prune": prune,
            "recovery": [],
        }

        if bool(prune["all_pruned"]):
            print(
                "[packet aggressive-prune branch] "
                f"th={threshold:.3f} G={prune['count_before']}->0 ALL_PRUNED",
                flush=True,
            )
            del branch
            torch.cuda.empty_cache()
            return result

        # checkpoint 0 = immediate post-prune render before any recovery step.
        if 0 in self.post_checkpoints:
            snap = self._snapshot(
                branch, cameras, threshold=threshold, post_iteration=0
            )
            result["recovery"].append(snap)
            print(
                "[packet aggressive-prune recovery] "
                f"th={threshold:.3f} post=0 "
                f"G={prune['count_before']}->{prune['count_after']} "
                f"pruned={100.0*prune['pruned_ratio']:.2f}% "
                f"PSNR(L/R/M)="
                f"{snap['stereo_metrics']['left']['psnr']:.3f}/"
                f"{snap['stereo_metrics']['right']['psnr']:.3f}/"
                f"{snap['stereo_metrics']['mean']['psnr']:.3f}",
                flush=True,
            )

        for offset in range(1, self.post_max_iters + 1):
            # Continue the GraphDECO schedule from the pre-prune trajectory.
            iteration = self.pre_iters + offset
            camera_index = 0 if self.supervision == "left" else (iteration - 1) % 2
            self._optimize_step(branch, cameras[camera_index], iteration)

            if offset in self.post_checkpoints:
                snap = self._snapshot(
                    branch,
                    cameras,
                    threshold=threshold,
                    post_iteration=offset,
                )
                result["recovery"].append(snap)
                mean = snap["stereo_metrics"]["mean"]
                pre_mean = pre_metrics["mean"]
                print(
                    "[packet aggressive-prune recovery] "
                    f"th={threshold:.3f} post={offset} "
                    f"G={snap['num_gaussians']} "
                    f"PSNR(L/R/M)="
                    f"{snap['stereo_metrics']['left']['psnr']:.3f}/"
                    f"{snap['stereo_metrics']['right']['psnr']:.3f}/"
                    f"{mean['psnr']:.3f} "
                    f"SSIM(M)={mean['ssim']:.4f} "
                    f"delta_vs_pre={mean['psnr'] - pre_mean['psnr']:+.3f}",
                    flush=True,
                )

        del branch
        torch.cuda.empty_cache()
        return result

    def _write_results(self, payload: dict[str, Any]) -> None:
        self.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        columns = [
            "threshold",
            "pre_iterations",
            "post_iteration",
            "gaussians_before",
            "gaussians_after",
            "pruned_count",
            "pruned_ratio",
            "pre_psnr",
            "pre_ssim",
            "post_psnr",
            "post_ssim",
            "post_psnr_delta_vs_pre",
            "left_psnr",
            "right_psnr",
            "opacity_mean",
        ]
        with self.output_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            for branch in payload["branches"]:
                prune = branch["prune"]
                pre = branch["pre_prune_stereo_metrics"]["mean"]
                for snap in branch["recovery"]:
                    stereo = snap["stereo_metrics"]
                    mean = stereo["mean"]
                    opacity = snap["opacity"]
                    writer.writerow(
                        {
                            "threshold": branch["threshold"],
                            "pre_iterations": branch["pre_iterations"],
                            "post_iteration": snap["post_iteration"],
                            "gaussians_before": prune["count_before"],
                            "gaussians_after": prune["count_after"],
                            "pruned_count": prune["pruned_count"],
                            "pruned_ratio": prune["pruned_ratio"],
                            "pre_psnr": pre["psnr"],
                            "pre_ssim": pre["ssim"],
                            "post_psnr": mean["psnr"],
                            "post_ssim": mean["ssim"],
                            "post_psnr_delta_vs_pre": mean["psnr"] - pre["psnr"],
                            "left_psnr": stereo["left"]["psnr"],
                            "right_psnr": stereo["right"]["psnr"],
                            "opacity_mean": opacity.get("mean", 0.0),
                        }
                    )

    def process_aggressive_sweep(self, result: Any) -> Any:
        packet = result.packet
        frame = int(packet.descriptor.frame_index)
        if frame != self.target_frame or self.completed:
            return result

        self.initialize()
        g = self._make_model(packet)
        cameras = self._make_cameras(result)
        input_count = int(g.get_xyz.shape[0])
        raw_metrics = self._evaluate_stereo(g, cameras)
        raw_opacity = self._opacity_stats(g)
        reset_stats = self._reset_opacity(g)
        reset_metrics = self._evaluate_stereo(g, cameras)

        print(
            "[packet aggressive-prune sweep start] "
            f"frame={frame} G={input_count} pre_iters={self.pre_iters} "
            f"thresholds={self.thresholds} post_max={self.post_max_iters} "
            f"post_checkpoints={self.post_checkpoints} supervision={self.supervision} "
            f"raw_PSNR={raw_metrics['mean']['psnr']:.3f} "
            f"reset_PSNR={reset_metrics['mean']['psnr']:.3f}",
            flush=True,
        )

        for iteration in range(1, self.pre_iters + 1):
            camera_index = 0 if self.supervision == "left" else (iteration - 1) % 2
            self._optimize_step(g, cameras[camera_index], iteration)

        pre_metrics = self._evaluate_stereo(g, cameras)
        pre_opacity = self._opacity_stats(g)
        print(
            "[packet aggressive-prune pre] "
            f"iter={self.pre_iters} PSNR(L/R/M)="
            f"{pre_metrics['left']['psnr']:.3f}/"
            f"{pre_metrics['right']['psnr']:.3f}/"
            f"{pre_metrics['mean']['psnr']:.3f} "
            f"SSIM(M)={pre_metrics['mean']['ssim']:.4f} "
            f"opacity_mean={pre_opacity.get('mean', 0.0):.5f}",
            flush=True,
        )

        branches = [
            self._run_branch(g, cameras, threshold, pre_metrics)
            for threshold in self.thresholds
        ]

        payload = {
            "frame_index": frame,
            "input_gaussians": input_count,
            "pre_iterations": self.pre_iters,
            "thresholds": list(self.thresholds),
            "post_max_iterations": self.post_max_iters,
            "post_checkpoints": list(self.post_checkpoints),
            "supervision": self.supervision,
            "reset_max_opacity": self.config.reset_max_opacity,
            "raw_stereo_metrics": raw_metrics,
            "raw_opacity": raw_opacity,
            "reset": reset_stats,
            "reset_stereo_metrics": reset_metrics,
            "pre_prune_stereo_metrics": pre_metrics,
            "pre_prune_opacity": pre_opacity,
            "branches": branches,
            "note": (
                "No densification is used. Each threshold branch is cloned from the "
                "same 150-step (or configured) stereo pre-optimization state, then "
                "actually pruned and optimized independently. The original ReSplat "
                "packet is handed downstream unchanged."
            ),
        }
        self._write_results(payload)
        self.completed = True
        print(
            f"[packet aggressive-prune sweep done] json={self.output_json} "
            f"csv={self.output_csv}",
            flush=True,
        )
        return result


def _peek_resolved_config() -> dict[str, Any]:
    return prefilter_v2.impl._peek_resolved_config()


def _install_sweep(resolved: dict[str, Any]) -> None:
    from async_pipeline.resplat_runtime import ResplatPacketGenerator

    target_frame = int(os.environ.get("PIPELINE_PACKET_AGGR_FRAME", "0"))
    pre_iters = int(os.environ.get("PIPELINE_PACKET_AGGR_PRE_ITERS", "150"))
    if pre_iters <= 0:
        raise ValueError("PIPELINE_PACKET_AGGR_PRE_ITERS must be positive")
    thresholds = _parse_thresholds(
        os.environ.get("PIPELINE_PACKET_AGGR_THRESHOLDS", "0.05,0.06,0.07,0.08,0.1")
    )
    post_max = int(os.environ.get("PIPELINE_PACKET_AGGR_POST_MAX_ITERS", "200"))
    if post_max <= 0:
        raise ValueError("PIPELINE_PACKET_AGGR_POST_MAX_ITERS must be positive")
    post_checkpoints = _parse_ints(
        os.environ.get(
            "PIPELINE_PACKET_AGGR_POST_CHECKPOINTS", "0,10,20,50,100,150,200"
        ),
        min_value=0,
        max_value=post_max,
    )
    reset_opacity = float(
        os.environ.get("PIPELINE_PACKET_AGGR_RESET_OPACITY", "0.01")
    )
    supervision = os.environ.get(
        "PIPELINE_PACKET_AGGR_SUPERVISION", "stereo"
    ).strip().lower()
    if supervision not in {"left", "stereo"}:
        raise ValueError("PIPELINE_PACKET_AGGR_SUPERVISION must be left or stereo")

    backend = resolved["backend"]
    config = LocalPacketPrefilterConfig(
        gs_repo=Path(resolved["paths"]["gs_repo"]),
        work_dir=Path(resolved["paths"]["work_dir"]),
        device=str(resolved["resplat_frontend"]["device"]),
        sh_degree=int(backend["sh_degree"]),
        iterations=pre_iters,
        post_prune_iterations=post_max,
        reset_max_opacity=reset_opacity,
        prune_min_opacity=min(thresholds),
        spatial_lr_scale=float(backend["spatial_lr_scale"]),
        white_background=bool(backend["white_background"]),
        antialiasing=bool(backend["antialiasing"]),
        log_every_iteration=False,
    )
    sweep = AggressivePruneRecoverySweep(
        config,
        dict(backend["optimization"]),
        target_frame=target_frame,
        pre_iters=pre_iters,
        thresholds=thresholds,
        post_max_iters=post_max,
        post_checkpoints=post_checkpoints,
        supervision=supervision,
    )

    original_infer = ResplatPacketGenerator.infer

    def infer_with_sweep(self: ResplatPacketGenerator, *args, **kwargs):
        result = original_infer(self, *args, **kwargs)
        return sweep.process_aggressive_sweep(result)

    infer_with_sweep._packet_aggressive_prune_recovery_patch = True  # type: ignore[attr-defined]
    ResplatPacketGenerator.infer = infer_with_sweep

    print(
        "[packet aggressive-prune sweep] enabled: "
        f"frame={target_frame}, pre={pre_iters}, thresholds={thresholds}, "
        f"post_max={post_max}, checkpoints={post_checkpoints}, "
        f"supervision={supervision}, densification=OFF",
        flush=True,
    )


if __name__ == "__main__":
    resolved = _peek_resolved_config()
    _install_sweep(resolved)
    repro.main()

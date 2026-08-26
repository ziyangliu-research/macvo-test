#!/usr/bin/env python3
"""Actual local-packet prune quality/compression sweep.

This diagnostic operates on one ReSplat packet only and never modifies the packet
handed to the downstream pipeline.  It is designed to answer the next question
after the read-only recovery curves:

    after opacity reset + stereo GraphDECO optimization, how much quality is
    actually lost when we prune at a chosen opacity threshold, and how much of
    that loss is recovered by a short post-prune optimization?

The main trajectory is continuous and unpruned.  At each configured checkpoint
(e.g. 100/150/200 iterations), the current Gaussian model and Adam state are
cloned once per threshold.  Each branch then performs a real opacity prune,
records immediate post-prune metrics, runs optional post-prune stereo optimization
(default 20 iterations), and records recovered metrics.  Therefore no threshold
or checkpoint contaminates any other branch.

Environment variables:
PIPELINE_PACKET_PRUNE_SWEEP_FRAME
    Target ReSplat frame. Default: 0.
PIPELINE_PACKET_PRUNE_SWEEP_MAX_ITERS
    Maximum pre-prune optimization iterations. Default: 200.
PIPELINE_PACKET_PRUNE_SWEEP_CHECKPOINTS
    Checkpoints to branch from. Default: 100,150,200.
PIPELINE_PACKET_PRUNE_SWEEP_THRESHOLDS
    Actual prune thresholds. Default: 0.02,0.03,0.05,0.1.
PIPELINE_PACKET_PRUNE_SWEEP_POST_ITERS
    Stereo optimizer steps after actual pruning. Default: 20.
PIPELINE_PACKET_PRUNE_SWEEP_RESET_OPACITY
    Opacity cap before pre-optimization. Default: 0.01.
PIPELINE_PACKET_PRUNE_SWEEP_SUPERVISION
    left or stereo for the main trajectory. Default: stereo.
"""
from __future__ import annotations

import copy
import csv
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn

import run_pipeline_execution_benchmark_repro as repro
import run_pipeline_execution_benchmark_packet_prefilter_v2 as prefilter_v2

LocalPacketPrefilter = prefilter_v2.impl.LocalPacketPrefilter
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


class ActualPacketPruneSweep(LocalPacketPrefilter):
    def __init__(
        self,
        config: LocalPacketPrefilterConfig,
        optimization: dict[str, Any],
        *,
        target_frame: int,
        max_iters: int,
        checkpoints: tuple[int, ...],
        thresholds: tuple[float, ...],
        post_iters: int,
        supervision: str,
    ) -> None:
        super().__init__(config, optimization)
        self.target_frame = int(target_frame)
        self.max_iters = int(max_iters)
        self.checkpoints = checkpoints
        self.thresholds = thresholds
        self.post_iters = int(post_iters)
        self.supervision = supervision
        self.completed = False

    @property
    def output_json(self) -> Path:
        return self.config.work_dir / "packet_actual_prune_sweep.json"

    @property
    def output_csv(self) -> Path:
        return self.config.work_dir / "packet_actual_prune_sweep.csv"

    def _clone_model_with_optimizer(self, source: Any) -> Any:
        """Clone Gaussian tensors and Adam state so branches are independent."""
        g = self.GaussianModel(self.config.sh_degree, self.opt.optimizer_type)
        g.spatial_lr_scale = float(source.spatial_lr_scale)
        g.active_sh_degree = int(source.active_sh_degree)

        def parameter(value: torch.Tensor) -> nn.Parameter:
            return nn.Parameter(value.detach().clone().requires_grad_(True))

        g._xyz = parameter(source._xyz)
        g._features_dc = parameter(source._features_dc)
        g._features_rest = parameter(source._features_rest)
        g._opacity = parameter(source._opacity)
        g._scaling = parameter(source._scaling)
        g._rotation = parameter(source._rotation)

        count = int(g._xyz.shape[0])
        g.max_radii2D = source.max_radii2D.detach().clone()
        g.xyz_gradient_accum = source.xyz_gradient_accum.detach().clone()
        g.denom = source.denom.detach().clone()
        source_tmp = getattr(source, "tmp_radii", None)
        if torch.is_tensor(source_tmp) and int(source_tmp.numel()) == count:
            g.tmp_radii = source_tmp.detach().clone()
        else:
            g.tmp_radii = torch.zeros(count, device=self.device, dtype=torch.float32)

        g.exposure_mapping = dict(getattr(source, "exposure_mapping", {}))
        g.pretrained_exposures = getattr(source, "pretrained_exposures", None)
        source_exposure = getattr(source, "_exposure", None)
        if torch.is_tensor(source_exposure):
            g._exposure = parameter(source_exposure)
        else:
            g._exposure = nn.Parameter(
                torch.eye(3, 4, device=self.device).unsqueeze(0).requires_grad_(True)
            )

        # Build optimizers/schedulers for the clone, then restore the exact Adam
        # moments and param-group learning rates from the checkpoint trajectory.
        g.training_setup(self.opt)
        g.optimizer.load_state_dict(copy.deepcopy(source.optimizer.state_dict()))
        g.max_radii2D = source.max_radii2D.detach().clone()
        g.xyz_gradient_accum = source.xyz_gradient_accum.detach().clone()
        g.denom = source.denom.detach().clone()
        return g

    @torch.no_grad()
    def _prune_threshold(self, g: Any, threshold: float) -> dict[str, Any]:
        before = int(g.get_xyz.shape[0])
        opacity = g.get_opacity.detach().reshape(-1)
        mask = opacity < float(threshold)
        pruned = int(mask.sum().item())
        keep = before - pruned
        stats = {
            "threshold": float(threshold),
            "count_before": before,
            "pruned_count": pruned,
            "pruned_ratio": 0.0 if before == 0 else float(pruned / before),
            "count_after": keep,
            "keep_ratio": 0.0 if before == 0 else float(keep / before),
            "all_pruned": bool(keep == 0),
        }
        if pruned > 0 and keep > 0:
            # GraphDECO prune_points expects tmp_radii to exist because the stock
            # call path normally comes through densify_and_prune().  Here it is
            # pure bookkeeping; no screen-size criterion is used.
            if not torch.is_tensor(getattr(g, "tmp_radii", None)):
                g.tmp_radii = torch.zeros(before, device=self.device)
            g.prune_points(mask)
            stats["count_after"] = int(g.get_xyz.shape[0])
        return stats

    def _run_post_prune(
        self,
        g: Any,
        cameras: list[Any],
        checkpoint: int,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for offset in range(1, self.post_iters + 1):
            iteration = checkpoint + offset
            camera_index = 0 if self.supervision == "left" else (iteration - 1) % 2
            train = self._optimize_step(g, cameras[camera_index], iteration)
            if offset in {1, 5, 10, self.post_iters}:
                rows.append(
                    {
                        "offset": int(offset),
                        "iteration": int(iteration),
                        "supervision_view": "left" if camera_index == 0 else "right",
                        "train": train,
                        "stereo_metrics": self._evaluate_stereo(g, cameras),
                        "opacity": self._opacity_stats(g),
                    }
                )
        return rows

    def _branch(
        self,
        source: Any,
        cameras: list[Any],
        checkpoint: int,
        threshold: float,
        pre_metrics: dict[str, Any],
    ) -> dict[str, Any]:
        branch = self._clone_model_with_optimizer(source)
        torch.cuda.synchronize(self.device)
        start = time.perf_counter()

        prune_stats = self._prune_threshold(branch, threshold)
        torch.cuda.synchronize(self.device)
        prune_sec = time.perf_counter() - start

        if bool(prune_stats["all_pruned"]):
            result = {
                "checkpoint": int(checkpoint),
                "threshold": float(threshold),
                "pre_prune_stereo_metrics": pre_metrics,
                "prune": prune_stats,
                "immediate_post_prune_stereo_metrics": None,
                "immediate_post_prune_opacity": None,
                "post_prune_optimization": [],
                "final_stereo_metrics": None,
                "final_opacity": None,
                "prune_sec": float(prune_sec),
                "post_prune_optimization_sec": 0.0,
            }
            del branch
            torch.cuda.empty_cache()
            return result

        immediate = self._evaluate_stereo(branch, cameras)
        immediate_opacity = self._opacity_stats(branch)

        torch.cuda.synchronize(self.device)
        recovery_start = time.perf_counter()
        post_log = self._run_post_prune(branch, cameras, checkpoint)
        torch.cuda.synchronize(self.device)
        recovery_sec = time.perf_counter() - recovery_start

        final = self._evaluate_stereo(branch, cameras)
        final_opacity = self._opacity_stats(branch)
        result = {
            "checkpoint": int(checkpoint),
            "threshold": float(threshold),
            "pre_prune_stereo_metrics": pre_metrics,
            "prune": prune_stats,
            "immediate_post_prune_stereo_metrics": immediate,
            "immediate_post_prune_opacity": immediate_opacity,
            "post_prune_optimization": post_log,
            "final_stereo_metrics": final,
            "final_opacity": final_opacity,
            "prune_sec": float(prune_sec),
            "post_prune_optimization_sec": float(recovery_sec),
        }
        del branch
        torch.cuda.empty_cache()
        return result

    def _write(self, payload: dict[str, Any]) -> None:
        self.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        columns = [
            "checkpoint",
            "threshold",
            "pre_psnr",
            "pre_ssim",
            "gaussians_before",
            "gaussians_after",
            "pruned_count",
            "pruned_ratio",
            "immediate_psnr",
            "immediate_ssim",
            "immediate_psnr_delta",
            "post_iters",
            "final_psnr",
            "final_ssim",
            "final_psnr_delta_vs_pre",
            "prune_sec",
            "post_optimization_sec",
        ]
        with self.output_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            for row in payload["branches"]:
                pre = row["pre_prune_stereo_metrics"]["mean"]
                immediate = row["immediate_post_prune_stereo_metrics"]
                final = row["final_stereo_metrics"]
                prune = row["prune"]
                immediate_mean = None if immediate is None else immediate["mean"]
                final_mean = None if final is None else final["mean"]
                writer.writerow(
                    {
                        "checkpoint": row["checkpoint"],
                        "threshold": row["threshold"],
                        "pre_psnr": pre["psnr"],
                        "pre_ssim": pre["ssim"],
                        "gaussians_before": prune["count_before"],
                        "gaussians_after": prune["count_after"],
                        "pruned_count": prune["pruned_count"],
                        "pruned_ratio": prune["pruned_ratio"],
                        "immediate_psnr": None if immediate_mean is None else immediate_mean["psnr"],
                        "immediate_ssim": None if immediate_mean is None else immediate_mean["ssim"],
                        "immediate_psnr_delta": None if immediate_mean is None else immediate_mean["psnr"] - pre["psnr"],
                        "post_iters": self.post_iters,
                        "final_psnr": None if final_mean is None else final_mean["psnr"],
                        "final_ssim": None if final_mean is None else final_mean["ssim"],
                        "final_psnr_delta_vs_pre": None if final_mean is None else final_mean["psnr"] - pre["psnr"],
                        "prune_sec": row["prune_sec"],
                        "post_optimization_sec": row["post_prune_optimization_sec"],
                    }
                )

    def process_sweep(self, result: Any) -> Any:
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
            "[packet actual-prune sweep start] "
            f"frame={frame} G={input_count} supervision={self.supervision} "
            f"max_iters={self.max_iters} checkpoints={self.checkpoints} "
            f"thresholds={self.thresholds} post_iters={self.post_iters} "
            f"raw_PSNR={raw_metrics['mean']['psnr']:.3f} "
            f"reset_PSNR={reset_metrics['mean']['psnr']:.3f}",
            flush=True,
        )

        main_curve: list[dict[str, Any]] = []
        branches: list[dict[str, Any]] = []

        for iteration in range(1, self.max_iters + 1):
            camera_index = 0 if self.supervision == "left" else (iteration - 1) % 2
            self._optimize_step(g, cameras[camera_index], iteration)

            if iteration in self.checkpoints:
                pre_metrics = self._evaluate_stereo(g, cameras)
                opacity = self._opacity_stats(g)
                main_curve.append(
                    {
                        "iteration": int(iteration),
                        "stereo_metrics": pre_metrics,
                        "opacity": opacity,
                        "num_gaussians": int(g.get_xyz.shape[0]),
                    }
                )
                print(
                    "[packet actual-prune checkpoint] "
                    f"iter={iteration} PSNR(L/R/M)="
                    f"{pre_metrics['left']['psnr']:.3f}/"
                    f"{pre_metrics['right']['psnr']:.3f}/"
                    f"{pre_metrics['mean']['psnr']:.3f} "
                    f"SSIM(M)={pre_metrics['mean']['ssim']:.4f} "
                    f"opacity_mean={float(opacity.get('mean', 0.0)):.5f}",
                    flush=True,
                )

                for threshold in self.thresholds:
                    row = self._branch(g, cameras, iteration, threshold, pre_metrics)
                    branches.append(row)
                    prune = row["prune"]
                    immediate = row["immediate_post_prune_stereo_metrics"]
                    final = row["final_stereo_metrics"]
                    immediate_psnr = None if immediate is None else immediate["mean"]["psnr"]
                    final_psnr = None if final is None else final["mean"]["psnr"]
                    print(
                        "[packet actual-prune branch] "
                        f"iter={iteration} th={threshold:.3f} "
                        f"G={prune['count_before']}->{prune['count_after']} "
                        f"pruned={100.0*prune['pruned_ratio']:.2f}% "
                        f"PSNR pre/immediate/post{self.post_iters}="
                        f"{pre_metrics['mean']['psnr']:.3f}/"
                        f"{('ALL_PRUNED' if immediate_psnr is None else f'{immediate_psnr:.3f}')}/"
                        f"{('N/A' if final_psnr is None else f'{final_psnr:.3f}')}",
                        flush=True,
                    )

        torch.cuda.synchronize(self.device)
        payload = {
            "frame_index": frame,
            "input_gaussians": input_count,
            "supervision": self.supervision,
            "max_iterations": self.max_iters,
            "checkpoint_iterations": list(self.checkpoints),
            "thresholds": list(self.thresholds),
            "post_prune_iterations": self.post_iters,
            "reset_max_opacity": self.config.reset_max_opacity,
            "raw_stereo_metrics": raw_metrics,
            "raw_opacity": raw_opacity,
            "reset": reset_stats,
            "reset_stereo_metrics": reset_metrics,
            "main_unpruned_curve": main_curve,
            "branches": branches,
            "note": (
                "The main recovery trajectory is never pruned. Each checkpoint/threshold "
                "branch clones Gaussian tensors and Adam state, performs an actual opacity "
                "prune, then optional post-prune optimization. The original ReSplat packet "
                "is handed downstream unchanged."
            ),
        }
        self._write(payload)
        self.completed = True
        print(
            f"[packet actual-prune sweep done] json={self.output_json} csv={self.output_csv}",
            flush=True,
        )
        return result


def _peek_resolved_config() -> dict[str, Any]:
    return prefilter_v2.impl._peek_resolved_config()


def _install_sweep(resolved: dict[str, Any]) -> None:
    from async_pipeline.resplat_runtime import ResplatPacketGenerator

    target_frame = int(os.environ.get("PIPELINE_PACKET_PRUNE_SWEEP_FRAME", "0"))
    max_iters = int(os.environ.get("PIPELINE_PACKET_PRUNE_SWEEP_MAX_ITERS", "200"))
    if max_iters <= 0:
        raise ValueError("PIPELINE_PACKET_PRUNE_SWEEP_MAX_ITERS must be positive")
    checkpoints = _parse_ints(
        os.environ.get("PIPELINE_PACKET_PRUNE_SWEEP_CHECKPOINTS", "100,150,200"),
        min_value=1,
        max_value=max_iters,
    )
    thresholds = _parse_thresholds(
        os.environ.get("PIPELINE_PACKET_PRUNE_SWEEP_THRESHOLDS", "0.02,0.03,0.05,0.1")
    )
    post_iters = int(os.environ.get("PIPELINE_PACKET_PRUNE_SWEEP_POST_ITERS", "20"))
    if post_iters < 0:
        raise ValueError("PIPELINE_PACKET_PRUNE_SWEEP_POST_ITERS must be non-negative")
    reset_opacity = float(
        os.environ.get("PIPELINE_PACKET_PRUNE_SWEEP_RESET_OPACITY", "0.01")
    )
    supervision = os.environ.get(
        "PIPELINE_PACKET_PRUNE_SWEEP_SUPERVISION", "stereo"
    ).strip().lower()
    if supervision not in {"left", "stereo"}:
        raise ValueError("PIPELINE_PACKET_PRUNE_SWEEP_SUPERVISION must be left or stereo")

    backend = resolved["backend"]
    config = LocalPacketPrefilterConfig(
        gs_repo=Path(resolved["paths"]["gs_repo"]),
        work_dir=Path(resolved["paths"]["work_dir"]),
        device=str(resolved["resplat_frontend"]["device"]),
        sh_degree=int(backend["sh_degree"]),
        iterations=max_iters,
        post_prune_iterations=post_iters,
        reset_max_opacity=reset_opacity,
        prune_min_opacity=min(thresholds),
        spatial_lr_scale=float(backend["spatial_lr_scale"]),
        white_background=bool(backend["white_background"]),
        antialiasing=bool(backend["antialiasing"]),
        log_every_iteration=False,
    )
    sweep = ActualPacketPruneSweep(
        config,
        dict(backend["optimization"]),
        target_frame=target_frame,
        max_iters=max_iters,
        checkpoints=checkpoints,
        thresholds=thresholds,
        post_iters=post_iters,
        supervision=supervision,
    )

    original_infer = ResplatPacketGenerator.infer

    def infer_with_sweep(self: ResplatPacketGenerator, *args, **kwargs):
        result = original_infer(self, *args, **kwargs)
        return sweep.process_sweep(result)

    infer_with_sweep._packet_actual_prune_sweep_patch = True  # type: ignore[attr-defined]
    ResplatPacketGenerator.infer = infer_with_sweep


if __name__ == "__main__":
    resolved = _peek_resolved_config()
    _install_sweep(resolved)
    repro.main()

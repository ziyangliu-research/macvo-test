#!/usr/bin/env python3
"""Fast integrated local-packet prefilter for sequence-scale benchmarks.

This runner is intended for 200-frame/full-pipeline evaluation rather than
per-iteration diagnostics.  It keeps the packet algorithm used in the local
experiments:

    ReSplat packet
      -> cap opacity
      -> alternating stereo pre-optimization
      -> opacity-only prune (no densification)
      -> alternating stereo post-prune optimization
      -> export surviving packet

Unlike the diagnostic LocalPacketPrefilter.process(), it performs no stereo
metric render after every optimizer step and does not call Tensor.item() in the
training loop.  CUDA synchronization is used only at phase boundaries for
meaningful timing.

The global backend is not modified by this script.  For the intended integrated
policy, pass ``--set backend.reset_new_packet_opacity=false`` so that the opacity
learned by the local packet optimizer is preserved at insertion.  Otherwise the
baseline backend will cap the filtered packet opacity again.

Environment variables
---------------------
PIPELINE_PACKET_FAST_PRE_ITERS
    Pre-prune stereo optimizer steps. Default: 150.
PIPELINE_PACKET_FAST_POST_ITERS
    Post-prune stereo optimizer steps. Default: 50.
PIPELINE_PACKET_FAST_RESET_OPACITY
    Physical opacity cap before local optimization. Default: 0.01.
PIPELINE_PACKET_FAST_PRUNE_THRESHOLD
    Local opacity prune threshold. Default: 0.07.
PIPELINE_PACKET_FAST_BOUNDARY_EVAL
    If true, render stereo metrics at raw/reset/pre-prune/post-prune/final
    boundaries. Default: false. Keep false for timing/FPS runs.
"""
from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path
from typing import Any

import torch

import run_pipeline_execution_benchmark_repro as repro
import run_pipeline_execution_benchmark_packet_prefilter_v2 as prefilter_v2

impl = prefilter_v2.impl
LocalPacketPrefilter = impl.LocalPacketPrefilter
LocalPacketPrefilterConfig = impl.LocalPacketPrefilterConfig


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class FastIntegratedPacketPrefilter(LocalPacketPrefilter):
    def __init__(self, *args, boundary_eval: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.boundary_eval = bool(boundary_eval)
        self.fast_log: list[dict[str, Any]] = []

    @property
    def fast_json_path(self) -> Path:
        return self.config.work_dir / "local_packet_prefilter_fast_log.json"

    @property
    def fast_csv_path(self) -> Path:
        return self.config.work_dir / "local_packet_prefilter_fast_summary.csv"

    def _train_step_no_host_sync(self, g: Any, camera: Any, iteration: int) -> None:
        assert self.background is not None
        g.update_learning_rate(iteration)
        render_pkg = self.render(
            camera,
            g,
            self.pipe,
            self.background,
            use_trained_exp=False,
            separate_sh=False,
        )
        image = render_pkg["render"]
        gt = camera.original_image
        ll1 = self.l1_loss(image, gt)
        ssim_value = self.ssim_fn(image, gt)
        loss = (
            (1.0 - self.opt.lambda_dssim) * ll1
            + self.opt.lambda_dssim * (1.0 - ssim_value)
        )
        loss.backward()
        with torch.no_grad():
            g.optimizer.step()
            g.optimizer.zero_grad(set_to_none=True)

    @torch.no_grad()
    def _actual_prune(self, g: Any) -> dict[str, Any]:
        before = int(g.get_xyz.shape[0])
        opacity = g.get_opacity.detach().reshape(-1)
        mask = opacity < float(self.config.prune_min_opacity)
        pruned = int(mask.sum().item())
        keep = before - pruned
        if keep <= 0:
            raise RuntimeError(
                "local packet prefilter would prune every Gaussian: "
                f"threshold={self.config.prune_min_opacity}, count={before}"
            )
        if pruned > 0:
            # GraphDECO prune_points assumes tmp_radii exists because its normal
            # call path is densify_and_prune(). The v2 compatibility patch also
            # initializes it, but keep this guard for robustness.
            tmp = getattr(g, "tmp_radii", None)
            if not torch.is_tensor(tmp) or int(tmp.numel()) != before:
                g.tmp_radii = torch.zeros(
                    before, device=self.device, dtype=torch.float32
                )
            g.prune_points(mask)
        after = int(g.get_xyz.shape[0])
        return {
            "count_before": before,
            "count_after": after,
            "pruned_count": pruned,
            "pruned_ratio": float(pruned / before) if before else 0.0,
        }

    def _maybe_eval(self, g: Any, cameras: list[Any]) -> dict[str, Any] | None:
        if not self.boundary_eval:
            return None
        return self._evaluate_stereo(g, cameras)

    def _write_fast_logs(self) -> None:
        self.fast_json_path.write_text(
            json.dumps(self.fast_log, indent=2), encoding="utf-8"
        )
        columns = [
            "frame_index",
            "input_gaussians",
            "output_gaussians",
            "pruned_count",
            "pruned_ratio",
            "pre_iterations",
            "post_iterations",
            "reset_max_opacity",
            "prune_threshold",
            "model_setup_sec",
            "pre_optimization_sec",
            "prune_sec",
            "post_optimization_sec",
            "export_sec",
            "prefilter_total_sec",
            "raw_mean_psnr",
            "pre_prune_mean_psnr",
            "immediate_post_prune_mean_psnr",
            "final_mean_psnr",
        ]
        with self.fast_csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            for row in self.fast_log:
                writer.writerow({key: row.get(key) for key in columns})

    def process_fast(self, result: Any) -> Any:
        self.initialize()
        packet = result.packet
        packet.validate()
        if packet.coordinate_frame != "left_camera_local":
            raise ValueError(
                "fast local packet prefilter requires left_camera_local packet"
            )

        total_start = time.perf_counter()
        torch.cuda.synchronize(self.device)
        setup_start = time.perf_counter()
        g = self._make_model(packet)
        cameras = self._make_cameras(result)
        input_count = int(g.get_xyz.shape[0])
        torch.cuda.synchronize(self.device)
        setup_sec = time.perf_counter() - setup_start

        raw_metrics = self._maybe_eval(g, cameras)
        reset_stats = self._reset_opacity(g)
        reset_metrics = self._maybe_eval(g, cameras)

        torch.cuda.synchronize(self.device)
        pre_start = time.perf_counter()
        for iteration in range(1, self.config.iterations + 1):
            camera_index = (iteration - 1) % 2
            self._train_step_no_host_sync(g, cameras[camera_index], iteration)
        torch.cuda.synchronize(self.device)
        pre_sec = time.perf_counter() - pre_start
        pre_metrics = self._maybe_eval(g, cameras)

        torch.cuda.synchronize(self.device)
        prune_start = time.perf_counter()
        prune_stats = self._actual_prune(g)
        torch.cuda.synchronize(self.device)
        prune_sec = time.perf_counter() - prune_start
        immediate_metrics = self._maybe_eval(g, cameras)

        torch.cuda.synchronize(self.device)
        post_start = time.perf_counter()
        for offset in range(1, self.config.post_prune_iterations + 1):
            iteration = self.config.iterations + offset
            camera_index = (offset - 1) % 2
            self._train_step_no_host_sync(g, cameras[camera_index], iteration)
        torch.cuda.synchronize(self.device)
        post_sec = time.perf_counter() - post_start
        final_metrics = self._maybe_eval(g, cameras)

        frame = int(packet.descriptor.frame_index)
        summary_for_packet = {
            "frame_index": frame,
            "input_gaussians": input_count,
            "output_gaussians": int(g.get_xyz.shape[0]),
            "pruned_count": int(prune_stats["pruned_count"]),
            "pruned_ratio": float(prune_stats["pruned_ratio"]),
            "iterations": int(self.config.iterations),
            "post_prune_iterations": int(self.config.post_prune_iterations),
            "reset_max_opacity": float(self.config.reset_max_opacity),
            "prune_min_opacity": float(self.config.prune_min_opacity),
            "fast_sequence_benchmark": True,
        }

        torch.cuda.synchronize(self.device)
        export_start = time.perf_counter()
        result.packet = self._export_packet(g, packet, summary_for_packet)
        torch.cuda.synchronize(self.device)
        export_sec = time.perf_counter() - export_start
        total_sec = time.perf_counter() - total_start

        def mean_psnr(value: dict[str, Any] | None) -> float | None:
            return None if value is None else float(value["mean"]["psnr"])

        row = {
            "frame_index": frame,
            "input_gaussians": input_count,
            "output_gaussians": int(prune_stats["count_after"]),
            "pruned_count": int(prune_stats["pruned_count"]),
            "pruned_ratio": float(prune_stats["pruned_ratio"]),
            "pre_iterations": int(self.config.iterations),
            "post_iterations": int(self.config.post_prune_iterations),
            "reset_max_opacity": float(self.config.reset_max_opacity),
            "prune_threshold": float(self.config.prune_min_opacity),
            "model_setup_sec": float(setup_sec),
            "pre_optimization_sec": float(pre_sec),
            "prune_sec": float(prune_sec),
            "post_optimization_sec": float(post_sec),
            "export_sec": float(export_sec),
            "prefilter_total_sec": float(total_sec),
            "raw_mean_psnr": mean_psnr(raw_metrics),
            "reset_mean_psnr": mean_psnr(reset_metrics),
            "pre_prune_mean_psnr": mean_psnr(pre_metrics),
            "immediate_post_prune_mean_psnr": mean_psnr(immediate_metrics),
            "final_mean_psnr": mean_psnr(final_metrics),
            "opacity_reset_before_mean": float(reset_stats["before_mean"]),
            "opacity_reset_after_mean": float(reset_stats["after_mean"]),
            "boundary_eval": self.boundary_eval,
        }
        self.fast_log.append(row)
        self._write_fast_logs()

        print(
            "[packet prefilter fast] "
            f"frame={frame} G={input_count}->{row['output_gaussians']} "
            f"pruned={100.0*row['pruned_ratio']:.2f}% "
            f"pre/post={row['pre_iterations']}/{row['post_iterations']} "
            f"th={row['prune_threshold']:.3f} "
            f"time(total/pre/prune/post)="
            f"{total_sec:.3f}/{pre_sec:.3f}/{prune_sec:.3f}/{post_sec:.3f}s",
            flush=True,
        )
        return result


def _install_fast_prefilter(resolved: dict[str, Any]) -> None:
    from async_pipeline.resplat_runtime import ResplatPacketGenerator

    original_infer = ResplatPacketGenerator.infer
    if getattr(original_infer, "_fast_local_packet_prefilter_patch", False):
        return

    backend = resolved["backend"]
    config = LocalPacketPrefilterConfig(
        gs_repo=Path(resolved["paths"]["gs_repo"]),
        work_dir=Path(resolved["paths"]["work_dir"]),
        device=str(resolved["resplat_frontend"]["device"]),
        sh_degree=int(backend["sh_degree"]),
        iterations=int(os.environ.get("PIPELINE_PACKET_FAST_PRE_ITERS", "150")),
        post_prune_iterations=int(
            os.environ.get("PIPELINE_PACKET_FAST_POST_ITERS", "50")
        ),
        reset_max_opacity=float(
            os.environ.get("PIPELINE_PACKET_FAST_RESET_OPACITY", "0.01")
        ),
        prune_min_opacity=float(
            os.environ.get("PIPELINE_PACKET_FAST_PRUNE_THRESHOLD", "0.07")
        ),
        spatial_lr_scale=float(backend["spatial_lr_scale"]),
        white_background=bool(backend["white_background"]),
        antialiasing=bool(backend["antialiasing"]),
        log_every_iteration=False,
    )
    prefilter = FastIntegratedPacketPrefilter(
        config,
        dict(backend["optimization"]),
        boundary_eval=_env_bool("PIPELINE_PACKET_FAST_BOUNDARY_EVAL", False),
    )

    if bool(backend.get("reset_new_packet_opacity", True)):
        print(
            "[packet prefilter fast WARNING] backend.reset_new_packet_opacity=true; "
            "the global backend will cap the locally optimized opacity again. "
            "For the integrated preserve-opacity policy, pass "
            "--set backend.reset_new_packet_opacity=false",
            flush=True,
        )

    def infer_with_fast_prefilter(self: ResplatPacketGenerator, *args, **kwargs):
        result = original_infer(self, *args, **kwargs)
        impl._materialize_packet_tensors(result)
        return prefilter.process_fast(result)

    infer_with_fast_prefilter._fast_local_packet_prefilter_patch = True  # type: ignore[attr-defined]
    ResplatPacketGenerator.infer = infer_with_fast_prefilter

    print(
        "[packet prefilter fast] enabled: "
        f"pre={config.iterations}, post={config.post_prune_iterations}, "
        f"reset={config.reset_max_opacity}, prune={config.prune_min_opacity}, "
        f"boundary_eval={prefilter.boundary_eval}, densification=off",
        flush=True,
    )


if __name__ == "__main__":
    resolved = impl._peek_resolved_config()
    _install_fast_prefilter(resolved)
    repro.main()

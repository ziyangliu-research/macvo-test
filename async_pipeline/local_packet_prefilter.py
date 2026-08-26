from __future__ import annotations

import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn

from .contracts import LocalGaussianPacket
from .geometry import quaternion_xyzw_to_wxyz


@dataclass(frozen=True)
class LocalPacketPrefilterConfig:
    """Configuration for ReSplat-packet-local GraphDECO filtering.

    This module intentionally operates before world alignment and before the
    incremental global backend.  The first experiment is deliberately simple:

    ReSplat packet -> opacity cap -> short stereo optimization -> opacity prune

    No local densification is performed.  The surviving packet is then handed to
    the unchanged global backend, which may apply its normal new-packet opacity
    reset again.  That keeps the first ablation focused on packet admission.
    """

    gs_repo: Path
    work_dir: Path
    device: str = "cuda:0"
    sh_degree: int = 3
    iterations: int = 10
    post_prune_iterations: int = 0
    reset_max_opacity: float = 0.01
    prune_min_opacity: float = 0.005
    spatial_lr_scale: float = 1.0
    white_background: bool = False
    antialiasing: bool = False
    log_every_iteration: bool = True

    def __post_init__(self) -> None:
        if self.iterations < 0:
            raise ValueError("local packet prefilter iterations must be non-negative")
        if self.post_prune_iterations < 0:
            raise ValueError("post-prune iterations must be non-negative")
        if not 0.0 < self.reset_max_opacity < 1.0:
            raise ValueError("reset_max_opacity must lie strictly in (0,1)")
        if not 0.0 <= self.prune_min_opacity < 1.0:
            raise ValueError("prune_min_opacity must lie in [0,1)")


class _LocalPacketCamera:
    """Minimal GraphDECO camera for one ReSplat stereo context view."""

    def __init__(
        self,
        *,
        image: torch.Tensor,
        K: torch.Tensor,
        Twc: torch.Tensor,
        get_projection_matrix,
        device: torch.device,
        name: str,
    ) -> None:
        image = image.detach().to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
        height, width = int(image.shape[-2]), int(image.shape[-1])
        K = K.detach().to(device=device, dtype=torch.float32).clone()

        # ReSplat normally stores normalized intrinsics in its context batch.
        # Convert to pixel units only for FoV construction when needed.
        if float(K[0, 0]) <= 2.0 and float(K[1, 1]) <= 2.0:
            K[0, 0] *= width
            K[0, 2] *= width
            K[1, 1] *= height
            K[1, 2] *= height

        fx, fy = float(K[0, 0]), float(K[1, 1])
        if fx <= 0 or fy <= 0:
            raise ValueError(f"invalid local packet camera focal lengths: fx={fx}, fy={fy}")

        self.FoVx = 2.0 * math.atan(width / (2.0 * fx))
        self.FoVy = 2.0 * math.atan(height / (2.0 * fy))
        self.image_width = width
        self.image_height = height
        self.original_image = image
        self.alpha_mask = torch.ones((1, height, width), device=device)
        self.image_name = name
        self.uid = 0
        self.colmap_id = 0
        self.data_device = device
        self.znear = 0.01
        self.zfar = 100.0

        Twc = Twc.detach().to(device=device, dtype=torch.float32)
        Tcw = torch.linalg.inv(Twc)
        self.world_view_transform = Tcw.transpose(0, 1).contiguous()
        self.projection_matrix = get_projection_matrix(
            znear=self.znear,
            zfar=self.zfar,
            fovX=self.FoVx,
            fovY=self.FoVy,
        ).transpose(0, 1).to(device)
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0)
            .bmm(self.projection_matrix.unsqueeze(0))
            .squeeze(0)
        )
        self.camera_center = self.world_view_transform.inverse()[3, :3]


class LocalPacketPrefilter:
    """Short stereo GraphDECO optimization and opacity pruning for each packet."""

    def __init__(
        self,
        config: LocalPacketPrefilterConfig,
        optimization: dict[str, Any] | SimpleNamespace,
    ) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.opt = (
            optimization
            if isinstance(optimization, SimpleNamespace)
            else SimpleNamespace(**optimization)
        )
        self._initialized = False
        self.render: Any = None
        self.GaussianModel: Any = None
        self.get_projection_matrix: Any = None
        self.psnr_fn: Any = None
        self.l1_loss: Any = None
        self.ssim_fn: Any = None
        self.background: torch.Tensor | None = None
        self.pipe: Any = None
        self.log: list[dict[str, Any]] = []

    @property
    def json_path(self) -> Path:
        return self.config.work_dir / "local_packet_prefilter_log.json"

    @property
    def csv_path(self) -> Path:
        return self.config.work_dir / "local_packet_prefilter_summary.csv"

    def initialize(self) -> None:
        if self._initialized:
            return
        repo = self.config.gs_repo.expanduser().resolve()
        if not repo.is_dir():
            raise FileNotFoundError(f"GraphDECO repository not found: {repo}")
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("local packet prefilter requires CUDA GraphDECO renderer")
        torch.cuda.set_device(self.device)
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))

        from gaussian_renderer import render
        from scene import GaussianModel
        from utils.graphics_utils import getProjectionMatrix
        from utils.image_utils import psnr
        from utils.loss_utils import l1_loss, ssim

        self.render = render
        self.GaussianModel = GaussianModel
        self.get_projection_matrix = getProjectionMatrix
        self.psnr_fn = psnr
        self.l1_loss = l1_loss
        self.ssim_fn = ssim
        self.background = torch.tensor(
            [1.0, 1.0, 1.0] if self.config.white_background else [0.0, 0.0, 0.0],
            dtype=torch.float32,
            device=self.device,
        )
        self.pipe = SimpleNamespace(
            convert_SHs_python=False,
            compute_cov3D_python=False,
            debug=False,
            antialiasing=self.config.antialiasing,
        )
        self.config.work_dir.mkdir(parents=True, exist_ok=True)
        self._initialized = True

    def _make_model(self, packet: LocalGaussianPacket):
        g = self.GaussianModel(self.config.sh_degree, self.opt.optimizer_type)
        g.spatial_lr_scale = float(self.config.spatial_lr_scale)

        means = packet.means.detach().to(self.device, dtype=torch.float32).contiguous()
        scales = packet.scales.detach().to(self.device, dtype=torch.float32).contiguous()
        rotations_xyzw = packet.rotations_xyzw.detach().to(
            self.device, dtype=torch.float32
        ).contiguous()
        harmonics = packet.harmonics.detach().to(self.device, dtype=torch.float32)
        opacity = packet.opacities.detach().reshape(-1, 1).to(
            self.device, dtype=torch.float32
        )

        target_coeffs = (self.config.sh_degree + 1) ** 2
        source_coeffs = int(harmonics.shape[-1])
        if source_coeffs < target_coeffs:
            pad = torch.zeros(
                harmonics.shape[0],
                3,
                target_coeffs - source_coeffs,
                device=self.device,
                dtype=harmonics.dtype,
            )
            harmonics = torch.cat([harmonics, pad], dim=-1)
        elif source_coeffs > target_coeffs:
            harmonics = harmonics[..., :target_coeffs]

        eps = 1e-6
        g._xyz = nn.Parameter(means.requires_grad_(True))
        g._features_dc = nn.Parameter(
            harmonics[:, :, 0].unsqueeze(1).contiguous().requires_grad_(True)
        )
        g._features_rest = nn.Parameter(
            harmonics[:, :, 1:].permute(0, 2, 1).contiguous().requires_grad_(True)
        )
        g._opacity = nn.Parameter(
            torch.logit(opacity.clamp(eps, 1.0 - eps)).contiguous().requires_grad_(True)
        )
        g._scaling = nn.Parameter(
            torch.log(scales.clamp_min(1e-8)).contiguous().requires_grad_(True)
        )
        g._rotation = nn.Parameter(
            quaternion_xyzw_to_wxyz(rotations_xyzw).contiguous().requires_grad_(True)
        )
        g.active_sh_degree = g.max_sh_degree

        count = int(g._xyz.shape[0])
        g.max_radii2D = torch.zeros(count, device=self.device)
        g.xyz_gradient_accum = torch.zeros((count, 1), device=self.device)
        g.denom = torch.zeros((count, 1), device=self.device)
        g.exposure_mapping = {}
        g.pretrained_exposures = None
        g._exposure = nn.Parameter(
            torch.eye(3, 4, device=self.device).unsqueeze(0).requires_grad_(True)
        )
        g.training_setup(self.opt)
        return g

    def _make_cameras(self, result: Any) -> list[_LocalPacketCamera]:
        packet = result.packet
        context = result.batch["context"]
        images = context["image"][0]
        if int(images.shape[0]) != 2:
            raise RuntimeError(
                f"local packet prefilter expects stereo context, got {tuple(images.shape)}"
            )

        cameras: list[_LocalPacketCamera] = []
        for index, side in enumerate(("left", "right")):
            cameras.append(
                _LocalPacketCamera(
                    image=images[index],
                    K=packet.context_intrinsics[index],
                    Twc=packet.context_extrinsics[index],
                    get_projection_matrix=self.get_projection_matrix,
                    device=self.device,
                    name=f"packet_{packet.descriptor.frame_index:06d}_{side}",
                )
            )
        return cameras

    def _render_metrics(self, g: Any, camera: _LocalPacketCamera) -> dict[str, float]:
        assert self.background is not None
        image = self.render(
            camera,
            g,
            self.pipe,
            self.background,
            use_trained_exp=False,
            separate_sh=False,
        )["render"].clamp(0.0, 1.0)
        gt = camera.original_image.clamp(0.0, 1.0)
        return {
            "l1": float(self.l1_loss(image, gt).mean().item()),
            "psnr": float(self.psnr_fn(image, gt).mean().item()),
            "ssim": float(self.ssim_fn(image, gt).mean().item()),
        }

    @torch.no_grad()
    def _evaluate_stereo(
        self, g: Any, cameras: list[_LocalPacketCamera]
    ) -> dict[str, Any]:
        left = self._render_metrics(g, cameras[0])
        right = self._render_metrics(g, cameras[1])
        mean = {
            key: 0.5 * (float(left[key]) + float(right[key]))
            for key in ("l1", "psnr", "ssim")
        }
        return {"left": left, "right": right, "mean": mean}

    @torch.no_grad()
    def _opacity_stats(self, g: Any) -> dict[str, float | int]:
        opacity = g.get_opacity.detach().reshape(-1)
        count = int(opacity.numel())
        if count == 0:
            return {"count": 0}
        return {
            "count": count,
            "mean": float(opacity.mean().item()),
            "min": float(opacity.min().item()),
            "max": float(opacity.max().item()),
            "lt_0.005_ratio": float((opacity < 0.005).float().mean().item()),
            "lt_0.01_ratio": float((opacity < 0.01).float().mean().item()),
            "lt_0.02_ratio": float((opacity < 0.02).float().mean().item()),
            "lt_0.05_ratio": float((opacity < 0.05).float().mean().item()),
            "lt_prune_ratio": float(
                (opacity < self.config.prune_min_opacity).float().mean().item()
            ),
        }

    @torch.no_grad()
    def _reset_opacity(self, g: Any) -> dict[str, Any]:
        before = g.get_opacity.detach().reshape(-1, 1)
        capped = torch.minimum(
            before,
            torch.full_like(before, self.config.reset_max_opacity),
        )
        eps = 1e-6
        g._opacity.copy_(torch.logit(capped.clamp(eps, 1.0 - eps)))
        return {
            "before_mean": float(before.mean().item()) if before.numel() else 0.0,
            "before_max": float(before.max().item()) if before.numel() else 0.0,
            "after_mean": float(capped.mean().item()) if capped.numel() else 0.0,
            "after_max": float(capped.max().item()) if capped.numel() else 0.0,
        }

    def _optimize_step(
        self,
        g: Any,
        camera: _LocalPacketCamera,
        iteration: int,
    ) -> dict[str, float]:
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
        return {
            "loss": float(loss.item()),
            "l1": float(ll1.item()),
            "ssim": float(ssim_value.item()),
        }

    @torch.no_grad()
    def _prune(self, g: Any) -> dict[str, int | float]:
        before = int(g.get_xyz.shape[0])
        mask = (g.get_opacity.reshape(-1) < self.config.prune_min_opacity)
        pruned = int(mask.sum().item())
        if pruned > 0:
            g.prune_points(mask)
        after = int(g.get_xyz.shape[0])
        return {
            "count_before": before,
            "count_after": after,
            "pruned_count": pruned,
            "pruned_ratio": 0.0 if before == 0 else float(pruned / before),
        }

    @torch.no_grad()
    def _export_packet(
        self,
        g: Any,
        source: LocalGaussianPacket,
        summary: dict[str, Any],
    ) -> LocalGaussianPacket:
        rotation_wxyz = g.get_rotation.detach()
        rotation_xyzw = torch.cat(
            [rotation_wxyz[:, 1:4], rotation_wxyz[:, 0:1]], dim=-1
        )
        harmonics = torch.cat(
            [
                g._features_dc.detach().permute(0, 2, 1),
                g._features_rest.detach().permute(0, 2, 1),
            ],
            dim=-1,
        )
        metadata = dict(source.metadata)
        metadata["local_packet_prefilter"] = summary

        packet = LocalGaussianPacket(
            descriptor=source.descriptor,
            means=g.get_xyz.detach().contiguous(),
            scales=g.get_scaling.detach().contiguous(),
            rotations_xyzw=rotation_xyzw.contiguous(),
            harmonics=harmonics.contiguous(),
            opacities=g.get_opacity.detach().reshape(-1).contiguous(),
            context_intrinsics=source.context_intrinsics.detach().to(self.device).contiguous(),
            context_extrinsics=source.context_extrinsics.detach().to(self.device).contiguous(),
            inference_sec=source.inference_sec,
            coordinate_frame=source.coordinate_frame,
            metadata=metadata,
        )
        packet.validate()

        # Preserve the original handoff semantics (normally pinned CPU).
        if source.means.device.type == "cpu":
            return packet.cpu(pin_memory=bool(source.means.is_pinned()))
        return packet

    def _write_logs(self) -> None:
        self.json_path.write_text(json.dumps(self.log, indent=2), encoding="utf-8")
        columns = [
            "frame_index",
            "input_gaussians",
            "output_gaussians",
            "pruned_count",
            "pruned_ratio",
            "raw_mean_psnr",
            "reset_mean_psnr",
            "pre_prune_mean_psnr",
            "post_prune_mean_psnr",
            "raw_mean_ssim",
            "post_prune_mean_ssim",
            "prefilter_total_sec",
            "optimization_sec",
            "prune_sec",
        ]
        with self.csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            for entry in self.log:
                writer.writerow({key: entry["summary"].get(key) for key in columns})

    def process(self, result: Any) -> Any:
        """Filter ``result.packet`` in-place and return the ReSplat result object."""
        self.initialize()
        packet = result.packet
        packet.validate()
        if packet.coordinate_frame != "left_camera_local":
            raise ValueError(
                "local packet prefilter requires ReSplat output_frame=left_camera_local"
            )

        total_start = time.perf_counter()
        g = self._make_model(packet)
        cameras = self._make_cameras(result)
        input_count = int(g.get_xyz.shape[0])

        raw_metrics = self._evaluate_stereo(g, cameras)
        raw_opacity = self._opacity_stats(g)
        reset_stats = self._reset_opacity(g)
        reset_metrics = self._evaluate_stereo(g, cameras)
        reset_opacity = self._opacity_stats(g)

        frame = int(packet.descriptor.frame_index)
        print(
            "[packet prefilter start] "
            f"frame={frame} G={input_count} "
            f"raw_psnr={raw_metrics['mean']['psnr']:.3f} "
            f"reset_psnr={reset_metrics['mean']['psnr']:.3f} "
            f"opacity={reset_stats['before_mean']:.5f}->{reset_stats['after_mean']:.5f}",
            flush=True,
        )

        iteration_log: list[dict[str, Any]] = []
        optimize_start = time.perf_counter()
        for iteration in range(1, self.config.iterations + 1):
            camera_index = (iteration - 1) % 2
            train_stats = self._optimize_step(g, cameras[camera_index], iteration)
            stereo = self._evaluate_stereo(g, cameras)
            opacity = self._opacity_stats(g)
            row = {
                "iteration": iteration,
                "supervision_view": "left" if camera_index == 0 else "right",
                "train": train_stats,
                "stereo_metrics_post_step": stereo,
                "opacity_post_step": opacity,
                "num_gaussians": int(g.get_xyz.shape[0]),
            }
            iteration_log.append(row)
            if self.config.log_every_iteration:
                print(
                    "[packet prefilter iter] "
                    f"frame={frame} iter={iteration:02d}/{self.config.iterations} "
                    f"view={row['supervision_view']} "
                    f"loss={train_stats['loss']:.6f} "
                    f"PSNR(L/R/M)="
                    f"{stereo['left']['psnr']:.3f}/"
                    f"{stereo['right']['psnr']:.3f}/"
                    f"{stereo['mean']['psnr']:.3f} "
                    f"SSIM(M)={stereo['mean']['ssim']:.4f} "
                    f"opacity_mean={opacity.get('mean', 0.0):.5f} "
                    f"lt_prune={100.0*float(opacity.get('lt_prune_ratio', 0.0)):.2f}%",
                    flush=True,
                )
        torch.cuda.synchronize(self.device)
        optimization_sec = time.perf_counter() - optimize_start

        pre_prune_metrics = self._evaluate_stereo(g, cameras)
        pre_prune_opacity = self._opacity_stats(g)
        prune_start = time.perf_counter()
        prune_stats = self._prune(g)
        torch.cuda.synchronize(self.device)
        prune_sec = time.perf_counter() - prune_start
        post_prune_metrics = self._evaluate_stereo(g, cameras)
        post_prune_opacity = self._opacity_stats(g)

        post_iteration_log: list[dict[str, Any]] = []
        for offset in range(1, self.config.post_prune_iterations + 1):
            iteration = self.config.iterations + offset
            camera_index = (offset - 1) % 2
            train_stats = self._optimize_step(g, cameras[camera_index], iteration)
            stereo = self._evaluate_stereo(g, cameras)
            opacity = self._opacity_stats(g)
            post_iteration_log.append(
                {
                    "iteration": iteration,
                    "supervision_view": "left" if camera_index == 0 else "right",
                    "train": train_stats,
                    "stereo_metrics_post_step": stereo,
                    "opacity_post_step": opacity,
                    "num_gaussians": int(g.get_xyz.shape[0]),
                }
            )

        final_metrics = self._evaluate_stereo(g, cameras)
        torch.cuda.synchronize(self.device)
        total_sec = time.perf_counter() - total_start

        summary = {
            "frame_index": frame,
            "input_gaussians": input_count,
            "output_gaussians": int(g.get_xyz.shape[0]),
            "pruned_count": int(prune_stats["pruned_count"]),
            "pruned_ratio": float(prune_stats["pruned_ratio"]),
            "raw_mean_psnr": float(raw_metrics["mean"]["psnr"]),
            "reset_mean_psnr": float(reset_metrics["mean"]["psnr"]),
            "pre_prune_mean_psnr": float(pre_prune_metrics["mean"]["psnr"]),
            "post_prune_mean_psnr": float(post_prune_metrics["mean"]["psnr"]),
            "raw_mean_ssim": float(raw_metrics["mean"]["ssim"]),
            "post_prune_mean_ssim": float(post_prune_metrics["mean"]["ssim"]),
            "prefilter_total_sec": float(total_sec),
            "optimization_sec": float(optimization_sec),
            "prune_sec": float(prune_sec),
            "iterations": int(self.config.iterations),
            "post_prune_iterations": int(self.config.post_prune_iterations),
            "reset_max_opacity": float(self.config.reset_max_opacity),
            "prune_min_opacity": float(self.config.prune_min_opacity),
        }
        entry = {
            "summary": summary,
            "raw_stereo_metrics": raw_metrics,
            "raw_opacity": raw_opacity,
            "reset": reset_stats,
            "reset_stereo_metrics": reset_metrics,
            "reset_opacity": reset_opacity,
            "iterations": iteration_log,
            "pre_prune_stereo_metrics": pre_prune_metrics,
            "pre_prune_opacity": pre_prune_opacity,
            "prune": prune_stats,
            "post_prune_stereo_metrics": post_prune_metrics,
            "post_prune_opacity": post_prune_opacity,
            "post_prune_optimization": post_iteration_log,
            "final_stereo_metrics": final_metrics,
        }
        self.log.append(entry)
        self._write_logs()

        print(
            "[packet prefilter prune] "
            f"frame={frame} G={input_count}->{summary['output_gaussians']} "
            f"pruned={summary['pruned_count']} "
            f"({100.0*summary['pruned_ratio']:.2f}%) "
            f"PSNR pre/post="
            f"{summary['pre_prune_mean_psnr']:.3f}/"
            f"{summary['post_prune_mean_psnr']:.3f} "
            f"total={total_sec:.3f}s",
            flush=True,
        )

        result.packet = self._export_packet(g, packet, summary)
        return result

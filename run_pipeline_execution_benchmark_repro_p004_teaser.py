#!/usr/bin/env python3
"""Full P004 qualitative teaser run with raw-packet and input-view artifacts.

This wrapper keeps the online mapping policy unchanged and adds visualization
artifacts only:

1. For every mapping timestamp, render the raw ReSplat local packet back into
   the left and right stereo input cameras with ReSplat's own decoder.
2. At the final online endpoint, save the complete GraphDECO Gaussian map and
   render that final map from every mapping/input camera.
3. Save an initial teaser view at the *first mapping camera pose*, but with the
   same wide 960x540 FoV used by the earlier P001 custom visualization.
4. Write the starting Twc to teaser_overview/probe_pose_current_relative.json
   so the existing headless web pose tuner can be used with --run_dir.

No post-hoc/global refinement is introduced.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torchvision.utils import save_image

import run_pipeline_execution_benchmark_repro_empty_safe as safe


_PACKET_ROWS: list[dict[str, Any]] = []


def _artifact_root() -> Path:
    raw = os.environ.get("PIPELINE_TEASER_ARTIFACT_ROOT")
    if not raw:
        raise RuntimeError("PIPELINE_TEASER_ARTIFACT_ROOT is required")
    root = Path(raw).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _psnr(pred: torch.Tensor, gt: torch.Tensor) -> float:
    mse = torch.mean((pred.float() - gt.float()) ** 2).clamp_min(1.0e-12)
    return float((-10.0 * torch.log10(mse)).item())


def _matrix4_list(value: torch.Tensor) -> list[list[float]]:
    return value.detach().double().cpu().tolist()


def _install_packet_artifacts() -> None:
    from async_pipeline.resplat_runtime import ResplatPacketGenerator

    original_infer = ResplatPacketGenerator.infer
    if getattr(original_infer, "_p004_teaser_packet_artifacts", False):
        return

    def infer_with_artifacts(self: ResplatPacketGenerator, frame_input, **kwargs):
        result = original_infer(self, frame_input, **kwargs)
        frame_index = int(frame_input.descriptor.frame_index)
        out_dir = _artifact_root() / "resplat_packet_renders" / f"{frame_index:06d}"
        out_dir.mkdir(parents=True, exist_ok=True)

        batch = result.batch
        assert self.model is not None

        # Render each stereo camera separately.  The left path is exactly the
        # target-left evaluation convention used by the ReSplat-only evaluator.
        with torch.inference_mode():
            left_out = self.model.decoder.forward(
                result.gaussians,
                batch["target"]["extrinsics"],
                batch["target"]["intrinsics"],
                batch["target"]["near"],
                batch["target"]["far"],
                tuple(int(x) for x in batch["target"]["image"].shape[-2:]),
                depth_mode=None,
            )
            right_out = self.model.decoder.forward(
                result.gaussians,
                batch["context"]["extrinsics"][:, 1:2],
                batch["context"]["intrinsics"][:, 1:2],
                batch["context"]["near"][:, 1:2],
                batch["context"]["far"][:, 1:2],
                tuple(int(x) for x in batch["context"]["image"].shape[-2:]),
                depth_mode=None,
            )
        torch.cuda.current_stream(self.device).synchronize()

        gt_left = batch["target"]["image"][0, 0].detach().clamp(0.0, 1.0)
        gt_right = batch["context"]["image"][0, 1].detach().clamp(0.0, 1.0)
        pred_left = left_out.color[0, 0].detach().clamp(0.0, 1.0)
        pred_right = right_out.color[0, 0].detach().clamp(0.0, 1.0)

        save_image(gt_left.cpu(), out_dir / "gt_left.png")
        save_image(gt_right.cpu(), out_dir / "gt_right.png")
        save_image(pred_left.cpu(), out_dir / "render_left.png")
        save_image(pred_right.cpu(), out_dir / "render_right.png")

        relative = (
            torch.linalg.inv(batch["context"]["extrinsics"][0, 0].detach().double().cpu())
            @ batch["context"]["extrinsics"][0, 1].detach().double().cpu()
        )
        row = {
            "frame_index": frame_index,
            "num_gaussians": int(result.packet.num_gaussians),
            "left_psnr_db": _psnr(pred_left, gt_left),
            "right_psnr_db": _psnr(pred_right, gt_right),
            "baseline_used_m": float(torch.linalg.vector_norm(relative[:3, 3]).item()),
            "T_left_from_right_used": _matrix4_list(relative),
            "resplat_image_shape_hw": list(batch["target"]["image"].shape[-2:]),
            "inference_sec": float(result.inference_sec),
        }
        (out_dir / "metrics.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
        _PACKET_ROWS.append(row)
        summary = {
            "num_packets": len(_PACKET_ROWS),
            "mean_left_psnr_db": sum(x["left_psnr_db"] for x in _PACKET_ROWS) / len(_PACKET_ROWS),
            "mean_right_psnr_db": sum(x["right_psnr_db"] for x in _PACKET_ROWS) / len(_PACKET_ROWS),
            "packets": _PACKET_ROWS,
        }
        summary_path = _artifact_root() / "resplat_packet_renders" / "summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(
            "[P004 ReSplat packet] "
            f"frame={frame_index:06d} G={row['num_gaussians']:,} "
            f"left={row['left_psnr_db']:.3f}dB right={row['right_psnr_db']:.3f}dB",
            flush=True,
        )
        return result

    infer_with_artifacts._p004_teaser_packet_artifacts = True  # type: ignore[attr-defined]
    ResplatPacketGenerator.infer = infer_with_artifacts


def _first_camera_twc(camera) -> torch.Tensor:
    # GraphDECO stores transposed world->camera in world_view_transform.
    tcw = camera.world_view_transform.transpose(0, 1).detach().double().cpu()
    return torch.linalg.inv(tcw)


def _make_wide_camera(self, twc: torch.Tensor):
    height = int(os.environ.get("PIPELINE_TEASER_RENDER_H", "540"))
    width = int(os.environ.get("PIPELINE_TEASER_RENDER_W", "960"))
    fx_norm = float(os.environ.get("PIPELINE_TEASER_FX_NORM", "0.25"))
    fy_norm = float(os.environ.get("PIPELINE_TEASER_FY_NORM", "0.35"))
    znear = float(os.environ.get("PIPELINE_TEASER_NEAR", "0.1"))
    zfar = float(os.environ.get("PIPELINE_TEASER_FAR", "50.0"))

    fx = fx_norm * width
    fy = fy_norm * height
    fovx = 2.0 * math.atan(width / (2.0 * fx))
    fovy = 2.0 * math.atan(height / (2.0 * fy))

    twc_device = twc.to(device=self.device, dtype=torch.float32)
    tcw = torch.linalg.inv(twc_device)
    world_view_transform = tcw.transpose(0, 1).contiguous()
    projection_matrix = self.get_projection_matrix(
        znear=znear, zfar=zfar, fovX=fovx, fovY=fovy
    ).transpose(0, 1).to(self.device)
    full_proj_transform = (
        world_view_transform.unsqueeze(0)
        .bmm(projection_matrix.unsqueeze(0))
        .squeeze(0)
    )
    return SimpleNamespace(
        FoVx=fovx,
        FoVy=fovy,
        image_width=width,
        image_height=height,
        world_view_transform=world_view_transform,
        projection_matrix=projection_matrix,
        full_proj_transform=full_proj_transform,
        camera_center=world_view_transform.inverse()[3, :3],
        znear=znear,
        zfar=zfar,
        uid=-1,
        colmap_id=-1,
        image_name="p004_teaser_first_view_wide",
        data_device=self.device,
    )


def _install_final_artifacts() -> None:
    from async_pipeline.backend_evaluation import BackendEvaluationMixin

    original_finalize = BackendEvaluationMixin._finalize_impl
    if getattr(original_finalize, "_p004_teaser_final_artifacts", False):
        return

    @torch.no_grad()
    def finalize_with_artifacts(self):
        root = _artifact_root()
        input_root = root / "final_map_input_renders"
        input_root.mkdir(parents=True, exist_ok=True)

        ply_path = None
        view_rows: list[dict[str, Any]] = []
        if self.gaussians is not None and int(self.gaussians.get_xyz.shape[0]) > 0:
            ply_path = self._save_point_cloud(self.global_iteration)

            for camera in self.train_cameras:
                frame_index = int(camera.frame_index)
                frame_dir = input_root / f"{frame_index:06d}"
                frame_dir.mkdir(parents=True, exist_ok=True)
                pred = self.render(
                    camera,
                    self.gaussians,
                    self.pipe,
                    self.background,
                    use_trained_exp=False,
                    separate_sh=False,
                )["render"].clamp(0.0, 1.0)
                gt = camera.original_image.clamp(0.0, 1.0)
                save_image(gt.detach().cpu(), frame_dir / "gt.png")
                save_image(pred.detach().cpu(), frame_dir / "render.png")
                row = {
                    "frame_index": frame_index,
                    "psnr_db": _psnr(pred, gt),
                    "width": int(camera.image_width),
                    "height": int(camera.image_height),
                }
                view_rows.append(row)
                (frame_dir / "metrics.json").write_text(
                    json.dumps(row, indent=2), encoding="utf-8"
                )

            # Initial teaser camera = exact first mapping-camera pose, but with
            # the old P001 wide FoV / 16:9 render canvas.
            if self.train_cameras:
                first_twc = _first_camera_twc(self.train_cameras[0])
                teaser_root = root / "teaser_overview"
                teaser_root.mkdir(parents=True, exist_ok=True)
                wide_camera = _make_wide_camera(self, first_twc)
                teaser = self.render(
                    wide_camera,
                    self.gaussians,
                    self.pipe,
                    self.background,
                    use_trained_exp=False,
                    separate_sh=False,
                )["render"].clamp(0.0, 1.0)
                save_image(teaser.detach().cpu(), teaser_root / "render.png")
                (teaser_root / "probe_pose_current_relative.json").write_text(
                    json.dumps(first_twc.tolist(), indent=2), encoding="utf-8"
                )
                meta = {
                    "purpose": "P004 teaser starting camera; first mapping pose with wide P001 FoV",
                    "frame_index": int(self.train_cameras[0].frame_index),
                    "Twc": first_twc.tolist(),
                    "render_width": int(os.environ.get("PIPELINE_TEASER_RENDER_W", "960")),
                    "render_height": int(os.environ.get("PIPELINE_TEASER_RENDER_H", "540")),
                    "fx_norm": float(os.environ.get("PIPELINE_TEASER_FX_NORM", "0.25")),
                    "fy_norm": float(os.environ.get("PIPELINE_TEASER_FY_NORM", "0.35")),
                    "cx_norm": 0.5,
                    "cy_norm": 0.5,
                    "num_gaussians": int(self.gaussians.get_xyz.shape[0]),
                }
                (teaser_root / "metadata.json").write_text(
                    json.dumps(meta, indent=2), encoding="utf-8"
                )

        manifest = {
            "stage": "final_online_endpoint_before_any_global_refinement",
            "num_gaussians": 0 if self.gaussians is None else int(self.gaussians.get_xyz.shape[0]),
            "global_iteration": int(self.global_iteration),
            "point_cloud_ply": None if ply_path is None else str(ply_path),
            "num_input_views": len(view_rows),
            "mean_input_view_psnr_db": (
                None if not view_rows else sum(x["psnr_db"] for x in view_rows) / len(view_rows)
            ),
            "views": view_rows,
        }
        (input_root / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        print(
            "[P004 teaser final] "
            f"G={manifest['num_gaussians']:,} input_views={len(view_rows)} "
            f"ply={manifest['point_cloud_ply']}",
            flush=True,
        )
        return original_finalize(self)

    finalize_with_artifacts._p004_teaser_final_artifacts = True  # type: ignore[attr-defined]
    BackendEvaluationMixin._finalize_impl = finalize_with_artifacts


if __name__ == "__main__":
    _install_packet_artifacts()
    _install_final_artifacts()
    safe.repro.main()

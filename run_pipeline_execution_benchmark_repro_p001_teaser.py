#!/usr/bin/env python3
"""Run the online P001 pipeline and save the legacy fixed overview view.

The canonical overview camera comes from ReSplat's
``assets/custom_poses/p001_render.json``.  That pose is an absolute TartanAir
Twc using OpenCV camera axes.  The current MAC-VO/GraphDECO map uses the first
selected left camera as the world origin, so before rendering we convert

    T_current_probe = inv(T_abs_left_frame0_cv) @ T_old_probe.

This wrapper performs visualization only; it adds no post-hoc optimization and
leaves the online mapping policy untouched.  The overview is rendered from the
final online Gaussian map before the normal backend finalize path returns.
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


def _quat_xyzw_to_matrix(values: list[float]) -> torch.Tensor:
    q = torch.tensor(values, dtype=torch.float64)
    q = q / torch.linalg.vector_norm(q).clamp_min(1e-12)
    x, y, z, w = q.unbind()
    return torch.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ]
    ).reshape(3, 3)


def _tartan_from_cv() -> torch.Tensor:
    out = torch.eye(4, dtype=torch.float64)
    out[:3, :3] = torch.tensor(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=torch.float64,
    )
    return out


def _load_first_absolute_left_pose_cv(path: Path) -> torch.Tensor:
    """Load first TartanAir tx ty tz qx qy qz qw row as absolute OpenCV Twc."""
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        values = [float(x) for x in line.replace(",", " ").split()[:7]]
        if len(values) != 7:
            continue
        tx, ty, tz, qx, qy, qz, qw = values
        twc_tartan_camera = torch.eye(4, dtype=torch.float64)
        twc_tartan_camera[:3, :3] = _quat_xyzw_to_matrix([qx, qy, qz, qw])
        twc_tartan_camera[:3, 3] = torch.tensor([tx, ty, tz], dtype=torch.float64)
        # Same camera-axis conversion used by both the legacy ReSplat fusion
        # utility and the current MAC-VO runtime.
        return twc_tartan_camera @ _tartan_from_cv()
    raise RuntimeError(f"no valid pose row found in {path}")


def _load_matrix4(path: Path) -> torch.Tensor:
    data: Any = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        for key in ("Twc", "extrinsics", "pose"):
            if key in data:
                data = data[key]
                break
    matrix = torch.tensor(data, dtype=torch.float64)
    if matrix.numel() == 16:
        matrix = matrix.reshape(4, 4)
    if tuple(matrix.shape) != (4, 4):
        raise ValueError(f"expected 4x4 pose in {path}, got {tuple(matrix.shape)}")
    return matrix


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _make_overview_camera(self, twc: torch.Tensor):
    height = _env_int("PIPELINE_TEASER_RENDER_H", 540)
    width = _env_int("PIPELINE_TEASER_RENDER_W", 960)
    fx_n = _env_float("PIPELINE_TEASER_FX_NORM", 0.25)
    fy_n = _env_float("PIPELINE_TEASER_FY_NORM", 0.35)
    cx_n = _env_float("PIPELINE_TEASER_CX_NORM", 0.5)
    cy_n = _env_float("PIPELINE_TEASER_CY_NORM", 0.5)
    if height <= 0 or width <= 0 or fx_n <= 0 or fy_n <= 0:
        raise ValueError("invalid teaser render size/intrinsics")
    # The current GraphDECO projection helper is centered-FoV based.  Preserve
    # the legacy custom view exactly where cx=cy=0.5; fail loudly for unsupported
    # off-center teaser intrinsics instead of silently changing the view.
    if abs(cx_n - 0.5) > 1e-8 or abs(cy_n - 0.5) > 1e-8:
        raise ValueError("P001 teaser renderer currently requires centered cx=cy=0.5")

    fx = fx_n * width
    fy = fy_n * height
    fovx = 2.0 * math.atan(width / (2.0 * fx))
    fovy = 2.0 * math.atan(height / (2.0 * fy))
    znear = _env_float("PIPELINE_TEASER_NEAR", 0.1)
    zfar = _env_float("PIPELINE_TEASER_FAR", 50.0)

    twc_d = twc.to(device=self.device, dtype=torch.float32)
    tcw = torch.linalg.inv(twc_d)
    world_view_transform = tcw.transpose(0, 1).contiguous()
    projection_matrix = self.get_projection_matrix(
        znear=znear,
        zfar=zfar,
        fovX=fovx,
        fovY=fovy,
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
        image_name="p001_teaser_overview",
        data_device=self.device,
    )


def _install_p001_teaser_render() -> None:
    from async_pipeline.backend_evaluation import BackendEvaluationMixin

    original_finalize = BackendEvaluationMixin._finalize_impl
    if getattr(original_finalize, "_p001_teaser_patch", False):
        return

    @torch.no_grad()
    def finalize_with_p001_teaser(self):
        probe_path = Path(
            os.environ.get(
                "PIPELINE_TEASER_PROBE_POSE_JSON",
                "../Resplat/assets/custom_poses/p001_render.json",
            )
        ).expanduser().resolve()
        gt_pose_path = Path(
            os.environ.get(
                "PIPELINE_TEASER_GT_POSE_FILE",
                "/home/shiyo/Desktop/Datasets/tartanair_v2/House/Data_easy/P001/pose_lcam_front.txt",
            )
        ).expanduser().resolve()

        if not probe_path.is_file():
            raise FileNotFoundError(f"legacy P001 probe pose not found: {probe_path}")
        if not gt_pose_path.is_file():
            raise FileNotFoundError(f"P001 left GT pose file not found: {gt_pose_path}")

        output_dir = self.output_dir / "teaser_overview"
        output_dir.mkdir(parents=True, exist_ok=True)

        old_probe_abs = _load_matrix4(probe_path)
        first_abs_cv = _load_first_absolute_left_pose_cv(gt_pose_path)
        current_probe = torch.linalg.inv(first_abs_cv) @ old_probe_abs

        metadata: dict[str, Any] = {
            "purpose": "qualitative teaser overview; no metric",
            "coordinate_conversion": "T_current_probe = inv(T_abs_left_frame0_cv) @ T_old_probe",
            "probe_pose_source": str(probe_path),
            "gt_pose_source": str(gt_pose_path),
            "old_probe_absolute_Twc": old_probe_abs.tolist(),
            "first_left_absolute_Twc_cv": first_abs_cv.tolist(),
            "current_first_frame_relative_probe_Twc": current_probe.tolist(),
            "render_height": _env_int("PIPELINE_TEASER_RENDER_H", 540),
            "render_width": _env_int("PIPELINE_TEASER_RENDER_W", 960),
            "intrinsics_norm": [
                _env_float("PIPELINE_TEASER_FX_NORM", 0.25),
                _env_float("PIPELINE_TEASER_FY_NORM", 0.35),
                _env_float("PIPELINE_TEASER_CX_NORM", 0.5),
                _env_float("PIPELINE_TEASER_CY_NORM", 0.5),
            ],
            "near": _env_float("PIPELINE_TEASER_NEAR", 0.1),
            "far": _env_float("PIPELINE_TEASER_FAR", 50.0),
            "num_train_packets": int(self.train_packet_count),
            "global_iteration": int(self.global_iteration),
            "num_gaussians": 0 if self.gaussians is None else int(self.gaussians.get_xyz.shape[0]),
        }

        if self.gaussians is None or int(self.gaussians.get_xyz.shape[0]) == 0:
            metadata["status"] = "EMPTY_MAP"
            (output_dir / "metadata.json").write_text(
                json.dumps(metadata, indent=2), encoding="utf-8"
            )
            print("[P001 teaser] final map is empty; no overview render saved", flush=True)
        else:
            camera = _make_overview_camera(self, current_probe)
            rendered = self.render(
                camera,
                self.gaussians,
                self.pipe,
                self.background,
                use_trained_exp=False,
                separate_sh=False,
            )["render"].clamp(0.0, 1.0)
            torch.cuda.current_stream(self.device).synchronize()
            render_path = output_dir / "render.png"
            save_image(rendered.detach().cpu(), render_path)
            (output_dir / "probe_pose_current_relative.json").write_text(
                json.dumps(current_probe.tolist(), indent=2), encoding="utf-8"
            )
            metadata["status"] = "OK"
            metadata["render_path"] = str(render_path)
            (output_dir / "metadata.json").write_text(
                json.dumps(metadata, indent=2), encoding="utf-8"
            )
            print(
                f"[P001 teaser] saved overview: {render_path} | "
                f"G={metadata['num_gaussians']} | "
                f"size={metadata['render_width']}x{metadata['render_height']}",
                flush=True,
            )

        return original_finalize(self)

    finalize_with_p001_teaser._p001_teaser_patch = True  # type: ignore[attr-defined]
    BackendEvaluationMixin._finalize_impl = finalize_with_p001_teaser


if __name__ == "__main__":
    _install_p001_teaser_render()
    safe.repro.main()

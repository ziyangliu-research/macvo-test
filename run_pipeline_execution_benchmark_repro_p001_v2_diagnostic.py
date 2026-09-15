#!/usr/bin/env python3
"""P001/TartanAir-V2 diagnostic wrapper for ReSplat + incremental 3DGS.

This runner is intentionally diagnostic-only. It keeps the normal serial online
pipeline unchanged, but saves enough intermediate evidence to localize a bad
P001 reconstruction:

1. Audit the actual V2 left/right GT pose files against the configured stereo
   rig (baseline magnitude, direction, and relative rotation).
2. For every selected mapping frame, render the *raw ReSplat packet* back into
   both stereo context cameras with ReSplat's own decoder. Save raw input,
   ReSplat-preprocessed input, packet renders, intrinsics/extrinsics and PSNR.
3. At the final online endpoint, save the complete GraphDECO Gaussian map and
   render it from every processed camera.
4. Reuse the previously validated custom P001 overview camera so the final map
   can also be inspected with the existing web pose tuner.

No post-hoc/global refinement is added by this wrapper.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torchvision.utils import save_image

import run_pipeline_execution_benchmark_repro_empty_safe as safe
import run_pipeline_execution_benchmark_repro_p001_teaser as teaser


_PACKET_METRICS: list[dict[str, Any]] = []


def _diag_root() -> Path:
    raw = os.environ.get("PIPELINE_P001_DIAG_OUTPUT")
    if not raw:
        raise RuntimeError("PIPELINE_P001_DIAG_OUTPUT is required")
    path = Path(raw).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _psnr(pred: torch.Tensor, gt: torch.Tensor) -> float:
    mse = torch.mean((pred.float() - gt.float()) ** 2).clamp_min(1.0e-12)
    return float((-10.0 * torch.log10(mse)).item())


def _tensor_stats(value: torch.Tensor) -> dict[str, Any]:
    x = value.detach().float().reshape(-1).cpu()
    finite = torch.isfinite(x)
    out: dict[str, Any] = {
        "shape": list(value.shape),
        "numel": int(x.numel()),
        "finite_fraction": float(finite.float().mean().item()) if x.numel() else 1.0,
    }
    if bool(finite.any()):
        xf = x[finite]
        out.update(
            {
                "min": float(xf.min().item()),
                "max": float(xf.max().item()),
                "mean": float(xf.mean().item()),
            }
        )
    return out


def _matrix_to_list(value: torch.Tensor) -> list[list[float]]:
    return value.detach().double().cpu().tolist()


def _rotation_angle_deg(R: torch.Tensor) -> float:
    trace = float(torch.trace(R).item())
    cosine = max(-1.0, min(1.0, (trace - 1.0) / 2.0))
    return math.degrees(math.acos(cosine))


def _load_tartan_pose_rows(path: Path) -> list[torch.Tensor]:
    rows: list[torch.Tensor] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            continue
        values = [float(x) for x in raw.replace(",", " ").split()[:7]]
        if len(values) != 7:
            continue
        tx, ty, tz, qx, qy, qz, qw = values
        T_tartan = torch.eye(4, dtype=torch.float64)
        T_tartan[:3, :3] = teaser._quat_xyzw_to_matrix([qx, qy, qz, qw])
        T_tartan[:3, 3] = torch.tensor([tx, ty, tz], dtype=torch.float64)
        # Convert camera axes exactly as the current P001 teaser/runtime contract.
        rows.append(T_tartan @ teaser._tartan_from_cv())
    if not rows:
        raise RuntimeError(f"no valid poses in {path}")
    return rows


def _write_v2_camera_audit() -> None:
    """Compare actual P001 V2 stereo poses with the runtime's fixed rig."""

    from async_pipeline.geometry import fixed_tartanair_stereo_rig_cv

    data_root = Path(
        os.environ.get(
            "PIPELINE_P001_DIAG_DATA_ROOT",
            "/home/shiyo/Desktop/Datasets/tartanair_v2/House/Data_easy/P001",
        )
    ).expanduser().resolve()
    max_frames = int(os.environ.get("PIPELINE_P001_DIAG_MAX_FRAMES", "20"))
    baseline = float(os.environ.get("PIPELINE_P001_DIAG_BASELINE", "0.25"))
    fx = float(os.environ.get("PIPELINE_P001_DIAG_FX", "320"))
    fy = float(os.environ.get("PIPELINE_P001_DIAG_FY", "320"))
    cx = float(os.environ.get("PIPELINE_P001_DIAG_CX", "320"))
    cy = float(os.environ.get("PIPELINE_P001_DIAG_CY", "320"))

    left_pose_path = data_root / "pose_lcam_front.txt"
    right_pose_path = data_root / "pose_rcam_front.txt"
    left_image_dir = data_root / "image_lcam_front"
    right_image_dir = data_root / "image_rcam_front"

    if not left_pose_path.is_file():
        raise FileNotFoundError(left_pose_path)
    if not right_pose_path.is_file():
        raise FileNotFoundError(
            f"P001 V2 stereo audit requires the actual right-camera pose file: {right_pose_path}"
        )

    left_rows = _load_tartan_pose_rows(left_pose_path)
    right_rows = _load_tartan_pose_rows(right_pose_path)
    count = min(max_frames, len(left_rows), len(right_rows))
    if count <= 0:
        raise RuntimeError("no overlapping left/right P001 poses")

    expected = fixed_tartanair_stereo_rig_cv(
        baseline, dtype=torch.float64
    ).cpu()
    expected_t = expected[:3, 3]

    per_frame: list[dict[str, Any]] = []
    trans_errors: list[float] = []
    rot_errors: list[float] = []
    baseline_norms: list[float] = []
    for i in range(count):
        actual = torch.linalg.inv(left_rows[i]) @ right_rows[i]
        t = actual[:3, 3]
        t_err = float(torch.linalg.vector_norm(t - expected_t).item())
        R_err = expected[:3, :3].transpose(0, 1) @ actual[:3, :3]
        r_err = _rotation_angle_deg(R_err)
        b_norm = float(torch.linalg.vector_norm(t).item())
        trans_errors.append(t_err)
        rot_errors.append(r_err)
        baseline_norms.append(b_norm)
        per_frame.append(
            {
                "frame_index": i,
                "actual_T_left_from_right_cv": _matrix_to_list(actual),
                "actual_translation_cv_m": [float(v) for v in t.tolist()],
                "actual_baseline_norm_m": b_norm,
                "translation_error_vs_runtime_m": t_err,
                "rotation_error_vs_runtime_deg": r_err,
            }
        )

    left_images = sorted(left_image_dir.glob("*.png"))
    right_images = sorted(right_image_dir.glob("*.png"))
    image_size = None
    if left_images:
        with Image.open(left_images[0]) as im:
            image_size = [int(im.width), int(im.height)]

    audit = {
        "dataset": "TartanAir V2 House/Data_easy/P001",
        "data_root": str(data_root),
        "checked_frames": count,
        "actual_image_size_wh": image_size,
        "stereo_image_counts": {
            "left": len(left_images),
            "right": len(right_images),
        },
        "runtime_camera_contract": {
            "width": 640,
            "height": 640,
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "stereo_baseline_m": baseline,
            "expected_T_left_from_right_cv": _matrix_to_list(expected),
        },
        "pose_file_measurement": {
            "baseline_norm_mean_m": sum(baseline_norms) / len(baseline_norms),
            "baseline_norm_min_m": min(baseline_norms),
            "baseline_norm_max_m": max(baseline_norms),
            "translation_error_max_m": max(trans_errors),
            "rotation_error_max_deg": max(rot_errors),
        },
        "per_frame": per_frame,
    }
    out = _diag_root() / "camera_audit.json"
    out.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(
        "[P001 V2 camera audit] "
        f"image={image_size} K=({fx},{fy},{cx},{cy}) "
        f"configured_baseline={baseline:.8f}m "
        f"pose_baseline_mean={audit['pose_file_measurement']['baseline_norm_mean_m']:.8f}m "
        f"max_t_err={max(trans_errors):.3e}m "
        f"max_R_err={max(rot_errors):.3e}deg",
        flush=True,
    )
    print(f"[P001 V2 camera audit] saved: {out}", flush=True)


def _install_resplat_packet_diagnostics() -> None:
    from async_pipeline.resplat_runtime import ResplatPacketGenerator

    original_infer = ResplatPacketGenerator.infer
    if getattr(original_infer, "_p001_packet_diagnostic_patch", False):
        return

    def infer_with_packet_diagnostics(self: ResplatPacketGenerator, frame_input, **kwargs):
        result = original_infer(self, frame_input, **kwargs)
        frame_index = int(frame_input.descriptor.frame_index)
        max_frames = int(os.environ.get("PIPELINE_P001_DIAG_MAX_FRAMES", "20"))
        if frame_index >= max_frames:
            return result

        packet_root = _diag_root() / "resplat_packet_renders" / f"{frame_index:06d}"
        packet_root.mkdir(parents=True, exist_ok=True)

        batch = result.batch
        h, w = (int(v) for v in batch["context"]["image"].shape[-2:])
        assert self.model is not None

        # In the formal serial runner ReSplat is pinned to CUDA's default stream.
        # Render with ReSplat's *own decoder* so this diagnostic does not involve
        # the GraphDECO packet conversion/backend at all.
        with torch.inference_mode():
            decoded = self.model.decoder.forward(
                result.gaussians,
                batch["context"]["extrinsics"],
                batch["context"]["intrinsics"],
                batch["context"]["near"],
                batch["context"]["far"],
                (h, w),
                depth_mode=None,
            )
        torch.cuda.current_stream(self.device).synchronize()

        pred = decoded.color[0].detach().clamp(0.0, 1.0)
        gt = batch["context"]["image"][0].detach().clamp(0.0, 1.0)
        if pred.shape[0] != 2 or gt.shape[0] != 2:
            raise RuntimeError(
                f"expected two stereo context views, pred={tuple(pred.shape)} gt={tuple(gt.shape)}"
            )

        # Save the original MAC-VO tensors as well as ReSplat's 320x320 domain.
        save_image(frame_input.left_image.detach().cpu().clamp(0.0, 1.0), packet_root / "input_left_raw.png")
        save_image(frame_input.right_image.detach().cpu().clamp(0.0, 1.0), packet_root / "input_right_raw.png")
        save_image(gt[0].cpu(), packet_root / "input_left_resplat.png")
        save_image(gt[1].cpu(), packet_root / "input_right_resplat.png")
        save_image(pred[0].cpu(), packet_root / "render_left_packet.png")
        save_image(pred[1].cpu(), packet_root / "render_right_packet.png")

        relative = (
            torch.linalg.inv(batch["context"]["extrinsics"][0, 0].detach().double().cpu())
            @ batch["context"]["extrinsics"][0, 1].detach().double().cpu()
        )
        entry: dict[str, Any] = {
            "frame_index": frame_index,
            "raw_left_shape": list(frame_input.left_image.shape),
            "raw_right_shape": list(frame_input.right_image.shape),
            "resplat_image_shape_hw": [h, w],
            "left_psnr_db": _psnr(pred[0], gt[0]),
            "right_psnr_db": _psnr(pred[1], gt[1]),
            "context_intrinsics": batch["context"]["intrinsics"][0].detach().double().cpu().tolist(),
            "context_extrinsics": batch["context"]["extrinsics"][0].detach().double().cpu().tolist(),
            "T_left_from_right_used_by_resplat": _matrix_to_list(relative),
            "baseline_used_norm_m": float(torch.linalg.vector_norm(relative[:3, 3]).item()),
            "num_gaussians": int(result.packet.means.shape[0]),
            "packet_stats": {
                "means": _tensor_stats(result.packet.means),
                "scales": _tensor_stats(result.packet.scales),
                "opacities": _tensor_stats(result.packet.opacities),
            },
            "inference_sec": float(result.inference_sec),
        }
        (packet_root / "metrics.json").write_text(
            json.dumps(entry, indent=2), encoding="utf-8"
        )
        _PACKET_METRICS.append(entry)
        summary = {
            "num_packets": len(_PACKET_METRICS),
            "mean_left_psnr_db": sum(x["left_psnr_db"] for x in _PACKET_METRICS) / len(_PACKET_METRICS),
            "mean_right_psnr_db": sum(x["right_psnr_db"] for x in _PACKET_METRICS) / len(_PACKET_METRICS),
            "packets": _PACKET_METRICS,
        }
        (_diag_root() / "resplat_packet_renders" / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        print(
            "[P001 ReSplat packet] "
            f"frame={frame_index:06d} G={entry['num_gaussians']} "
            f"left={entry['left_psnr_db']:.3f}dB "
            f"right={entry['right_psnr_db']:.3f}dB "
            f"baseline={entry['baseline_used_norm_m']:.8f}m",
            flush=True,
        )
        return result

    infer_with_packet_diagnostics._p001_packet_diagnostic_patch = True  # type: ignore[attr-defined]
    ResplatPacketGenerator.infer = infer_with_packet_diagnostics


def _install_final_map_diagnostics() -> None:
    from async_pipeline.backend_evaluation import BackendEvaluationMixin

    original_finalize = BackendEvaluationMixin._finalize_impl
    if getattr(original_finalize, "_p001_final_map_diagnostic_patch", False):
        return

    @torch.no_grad()
    def finalize_with_map_diagnostics(self):
        root = _diag_root()
        render_root = root / "final_map_renders"
        render_root.mkdir(parents=True, exist_ok=True)

        entries: list[dict[str, Any]] = []
        ply_path = None
        if self.gaussians is not None and int(self.gaussians.get_xyz.shape[0]) > 0:
            ply_path = self._save_point_cloud(self.global_iteration)
            for camera in self.train_cameras:
                frame_index = int(camera.frame_index)
                frame_dir = render_root / f"{frame_index:06d}"
                frame_dir.mkdir(parents=True, exist_ok=True)
                rendered = self.render(
                    camera,
                    self.gaussians,
                    self.pipe,
                    self.background,
                    use_trained_exp=False,
                    separate_sh=False,
                )["render"].clamp(0.0, 1.0)
                gt = camera.original_image.clamp(0.0, 1.0)
                torch.cuda.current_stream(self.device).synchronize()
                save_image(gt.detach().cpu(), frame_dir / "gt.png")
                save_image(rendered.detach().cpu(), frame_dir / "render.png")
                entry = {
                    "frame_index": frame_index,
                    "psnr_db": _psnr(rendered, gt),
                    "width": int(camera.image_width),
                    "height": int(camera.image_height),
                }
                entries.append(entry)
                (frame_dir / "metrics.json").write_text(
                    json.dumps(entry, indent=2), encoding="utf-8"
                )

        manifest = {
            "stage": "final_online_endpoint_before_any_global_refinement",
            "num_gaussians": 0 if self.gaussians is None else int(self.gaussians.get_xyz.shape[0]),
            "global_iteration": int(self.global_iteration),
            "point_cloud_ply": None if ply_path is None else str(ply_path),
            "num_rendered_mapping_views": len(entries),
            "mean_mapping_psnr_db": (
                None if not entries else sum(e["psnr_db"] for e in entries) / len(entries)
            ),
            "views": entries,
        }
        (render_root / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        print(
            "[P001 final-map diagnostic] "
            f"G={manifest['num_gaussians']} views={len(entries)} "
            f"mean_psnr={manifest['mean_mapping_psnr_db']} ply={manifest['point_cloud_ply']}",
            flush=True,
        )
        return original_finalize(self)

    finalize_with_map_diagnostics._p001_final_map_diagnostic_patch = True  # type: ignore[attr-defined]
    BackendEvaluationMixin._finalize_impl = finalize_with_map_diagnostics


if __name__ == "__main__":
    _write_v2_camera_audit()
    _install_resplat_packet_diagnostics()
    _install_final_map_diagnostics()
    # Install this last so the validated legacy custom overview is rendered first,
    # then our final-map diagnostics run, then the normal backend finalize executes.
    teaser._install_p001_teaser_render()
    safe.repro.main()

#!/usr/bin/env python3
"""Interactive pose tuner for the saved P001 final Gaussian map.

Starts from the already-converted legacy custom pose saved by
``run_p001_teaser_overview.sh`` and lets the user manually move/rotate the
camera while repeatedly rendering the same final Online Gaussian map.

Default inputs:
  outputs/p001_teaser_overview/full/incremental_P001_teaser_full/
    point_cloud/iteration_*/point_cloud.ply
    teaser_overview/probe_pose_current_relative.json

The tool never re-runs MAC-VO, ReSplat, or online optimization.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch


def _latest_ply(run_dir: Path) -> Path:
    candidates: list[tuple[int, Path]] = []
    for p in (run_dir / "point_cloud").glob("iteration_*/point_cloud.ply"):
        try:
            it = int(p.parent.name.split("_", 1)[1])
        except Exception:
            continue
        candidates.append((it, p))
    if not candidates:
        raise FileNotFoundError(f"no point_cloud/iteration_*/point_cloud.ply under {run_dir}")
    return max(candidates, key=lambda x: x[0])[1]


def _load_pose(path: Path) -> torch.Tensor:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        for key in ("Twc", "pose", "extrinsics"):
            if key in data:
                data = data[key]
                break
    pose = torch.tensor(data, dtype=torch.float64)
    if pose.numel() == 16:
        pose = pose.reshape(4, 4)
    if tuple(pose.shape) != (4, 4):
        raise ValueError(f"expected 4x4 pose in {path}, got {tuple(pose.shape)}")
    return pose


def _orthonormalize(r: torch.Tensor) -> torch.Tensor:
    u, _, vh = torch.linalg.svd(r)
    out = u @ vh
    if torch.det(out) < 0:
        u[:, -1] *= -1
        out = u @ vh
    return out


def _rot_x(a: float) -> torch.Tensor:
    c, s = math.cos(a), math.sin(a)
    return torch.tensor([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=torch.float64)


def _rot_y(a: float) -> torch.Tensor:
    c, s = math.cos(a), math.sin(a)
    return torch.tensor([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=torch.float64)


def _rot_z(a: float) -> torch.Tensor:
    c, s = math.cos(a), math.sin(a)
    return torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64)


def _translate_local(pose: torch.Tensor, dx: float, dy: float, dz: float) -> None:
    delta = torch.tensor([dx, dy, dz], dtype=torch.float64)
    pose[:3, 3] += pose[:3, :3] @ delta


def _rotate_local(pose: torch.Tensor, r_local: torch.Tensor) -> None:
    pose[:3, :3] = _orthonormalize(pose[:3, :3] @ r_local)


def _make_camera(get_projection_matrix, pose: torch.Tensor, device: torch.device,
                 width: int, height: int, fx_norm: float, fy_norm: float,
                 focal_scale: float, znear: float, zfar: float):
    fx = fx_norm * width * focal_scale
    fy = fy_norm * height * focal_scale
    fovx = 2.0 * math.atan(width / (2.0 * fx))
    fovy = 2.0 * math.atan(height / (2.0 * fy))

    twc = pose.to(device=device, dtype=torch.float32)
    tcw = torch.linalg.inv(twc)
    world_view_transform = tcw.transpose(0, 1).contiguous()
    projection_matrix = get_projection_matrix(
        znear=znear, zfar=zfar, fovX=fovx, fovY=fovy
    ).transpose(0, 1).to(device)
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
        image_name="p001_teaser_pose_tuner",
        data_device=device,
    )


def _tensor_to_bgr(image: torch.Tensor) -> np.ndarray:
    rgb = (
        image.detach().clamp(0.0, 1.0)
        .permute(1, 2, 0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .cpu()
        .numpy()
    )
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _write_pose(path: Path, pose: torch.Tensor) -> None:
    path.write_text(json.dumps(pose.tolist(), indent=2), encoding="utf-8")


def _save_state(output_dir: Path, stem: str, image_bgr: np.ndarray,
                pose: torch.Tensor, args, focal_scale: float) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_dir / f"{stem}.png"), image_bgr)
    _write_pose(output_dir / f"{stem}_pose.json", pose)
    meta = {
        "Twc": pose.tolist(),
        "width": args.width,
        "height": args.height,
        "fx_norm_base": args.fx_norm,
        "fy_norm_base": args.fy_norm,
        "focal_scale": focal_scale,
        "effective_fx_norm": args.fx_norm * focal_scale,
        "effective_fy_norm": args.fy_norm * focal_scale,
        "cx_norm": 0.5,
        "cy_norm": 0.5,
        "znear": args.znear,
        "zfar": args.zfar,
    }
    (output_dir / f"{stem}_camera.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )


def _print_help() -> None:
    print(
        "\nControls (OpenCV camera axes: +x right, +y down, +z forward):\n"
        "  w / s : forward / backward\n"
        "  a / d : left / right\n"
        "  r / f : up / down\n"
        "  j / l : yaw left / right\n"
        "  i / k : pitch up / down\n"
        "  u / o : roll CCW / CW\n"
        "  [ / ] : wider FOV / narrower FOV\n"
        "  1 / 2 : halve / double translation step\n"
        "  3 / 4 : halve / double rotation step\n"
        "  0     : reset to starting pose + FOV\n"
        "  p     : save numbered candidate PNG + pose JSON\n"
        "  Enter : save FINAL PNG + pose JSON and exit\n"
        "  h     : print this help\n"
        "  Esc   : exit without changing FINAL\n"
    )


def _load_gaussians(gs_repo: Path, ply: Path, device: torch.device):
    if str(gs_repo) not in sys.path:
        sys.path.insert(0, str(gs_repo))
    from gaussian_renderer import render
    from scene import GaussianModel
    from utils.graphics_utils import getProjectionMatrix

    gaussians = GaussianModel(3, "default")
    try:
        gaussians.load_ply(str(ply), use_train_test_exp=False)
    except TypeError:
        gaussians.load_ply(str(ply))
    gaussians = gaussians.to(device) if hasattr(gaussians, "to") else gaussians

    pipe = SimpleNamespace(
        convert_SHs_python=False,
        compute_cov3D_python=False,
        debug=False,
        antialiasing=False,
    )
    background = torch.zeros(3, dtype=torch.float32, device=device)
    return gaussians, render, getProjectionMatrix, pipe, background


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_dir",
        type=Path,
        default=Path("outputs/p001_teaser_overview/full/incremental_P001_teaser_full"),
    )
    parser.add_argument("--ply", type=Path, default=None)
    parser.add_argument("--start_pose", type=Path, default=None)
    parser.add_argument("--gs_repo", type=Path, default=Path("../gaussian-splatting"))
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--fx_norm", type=float, default=0.25)
    parser.add_argument("--fy_norm", type=float, default=0.35)
    parser.add_argument("--translation_step", type=float, default=0.10)
    parser.add_argument("--rotation_step_deg", type=float, default=2.0)
    parser.add_argument("--fov_step", type=float, default=1.05)
    parser.add_argument("--znear", type=float, default=0.1)
    parser.add_argument("--zfar", type=float, default=50.0)
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    ply = args.ply.expanduser().resolve() if args.ply else _latest_ply(run_dir)
    start_pose_path = (
        args.start_pose.expanduser().resolve()
        if args.start_pose
        else (run_dir / "teaser_overview" / "probe_pose_current_relative.json").resolve()
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else (run_dir / "teaser_pose_tuning").resolve()
    )
    gs_repo = args.gs_repo.expanduser().resolve()

    if not ply.is_file():
        raise FileNotFoundError(ply)
    if not start_pose_path.is_file():
        raise FileNotFoundError(start_pose_path)
    if not gs_repo.is_dir():
        raise FileNotFoundError(gs_repo)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    start_pose = _load_pose(start_pose_path)
    pose = start_pose.clone()
    focal_scale = 1.0
    translation_step = float(args.translation_step)
    rotation_step = math.radians(float(args.rotation_step_deg))

    print(f"[pose-tuner] PLY       : {ply}")
    print(f"[pose-tuner] start pose: {start_pose_path}")
    print(f"[pose-tuner] output    : {output_dir}")
    print("[pose-tuner] loading final Gaussian map ...", flush=True)
    gaussians, render, get_projection_matrix, pipe, background = _load_gaussians(
        gs_repo, ply, device
    )
    print(f"[pose-tuner] Gaussians : {int(gaussians.get_xyz.shape[0])}")
    _print_help()

    candidate_idx = 0
    window = "P001 teaser pose tuner"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    def render_current() -> np.ndarray:
        camera = _make_camera(
            get_projection_matrix,
            pose,
            device,
            args.width,
            args.height,
            args.fx_norm,
            args.fy_norm,
            focal_scale,
            args.znear,
            args.zfar,
        )
        with torch.inference_mode():
            image = render(
                camera,
                gaussians,
                pipe,
                background,
                use_trained_exp=False,
                separate_sh=False,
            )["render"]
        torch.cuda.synchronize(device)
        return _tensor_to_bgr(image)

    image_bgr = render_current()
    while True:
        display = image_bgr.copy()
        cv2.putText(
            display,
            f"move={translation_step:.4f}m  rot={math.degrees(rotation_step):.3f}deg  focal={focal_scale:.4f}",
            (12, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.imshow(window, display)
        key = cv2.waitKey(0) & 0xFF
        rerender = True

        if key == 27:  # Esc
            print("[pose-tuner] exit without saving FINAL")
            break
        if key in (10, 13):  # Enter
            _save_state(output_dir, "final", image_bgr, pose, args, focal_scale)
            print(f"[pose-tuner] FINAL saved under: {output_dir}")
            break
        if key == ord("h"):
            _print_help()
            rerender = False
        elif key == ord("p"):
            candidate_idx += 1
            stem = f"candidate_{candidate_idx:03d}"
            _save_state(output_dir, stem, image_bgr, pose, args, focal_scale)
            print(f"[pose-tuner] saved {stem}")
            rerender = False
        elif key == ord("0"):
            pose = start_pose.clone()
            focal_scale = 1.0
            print("[pose-tuner] reset")
        elif key == ord("1"):
            translation_step *= 0.5
            print(f"[pose-tuner] translation step = {translation_step:.6f} m")
            rerender = False
        elif key == ord("2"):
            translation_step *= 2.0
            print(f"[pose-tuner] translation step = {translation_step:.6f} m")
            rerender = False
        elif key == ord("3"):
            rotation_step *= 0.5
            print(f"[pose-tuner] rotation step = {math.degrees(rotation_step):.6f} deg")
            rerender = False
        elif key == ord("4"):
            rotation_step *= 2.0
            print(f"[pose-tuner] rotation step = {math.degrees(rotation_step):.6f} deg")
            rerender = False
        elif key == ord("["):
            focal_scale /= args.fov_step
        elif key == ord("]"):
            focal_scale *= args.fov_step
        elif key == ord("w"):
            _translate_local(pose, 0.0, 0.0, translation_step)
        elif key == ord("s"):
            _translate_local(pose, 0.0, 0.0, -translation_step)
        elif key == ord("a"):
            _translate_local(pose, -translation_step, 0.0, 0.0)
        elif key == ord("d"):
            _translate_local(pose, translation_step, 0.0, 0.0)
        elif key == ord("r"):
            _translate_local(pose, 0.0, -translation_step, 0.0)
        elif key == ord("f"):
            _translate_local(pose, 0.0, translation_step, 0.0)
        elif key == ord("j"):
            _rotate_local(pose, _rot_y(-rotation_step))
        elif key == ord("l"):
            _rotate_local(pose, _rot_y(rotation_step))
        elif key == ord("i"):
            _rotate_local(pose, _rot_x(rotation_step))
        elif key == ord("k"):
            _rotate_local(pose, _rot_x(-rotation_step))
        elif key == ord("u"):
            _rotate_local(pose, _rot_z(-rotation_step))
        elif key == ord("o"):
            _rotate_local(pose, _rot_z(rotation_step))
        else:
            rerender = False

        if rerender:
            image_bgr = render_current()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

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

UI modes:
  --ui auto    Use OpenCV GUI when a display is available; otherwise web UI.
  --ui opencv  Force the original OpenCV HighGUI window.
  --ui web     Headless browser UI, intended for VS Code Remote SSH.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlparse

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
    return torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=torch.float64
    )


def _rot_y(a: float) -> torch.Tensor:
    c, s = math.cos(a), math.sin(a)
    return torch.tensor(
        [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=torch.float64
    )


def _rot_z(a: float) -> torch.Tensor:
    c, s = math.cos(a), math.sin(a)
    return torch.tensor(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64
    )


def _translate_local(pose: torch.Tensor, dx: float, dy: float, dz: float) -> None:
    delta = torch.tensor([dx, dy, dz], dtype=torch.float64)
    pose[:3, 3] += pose[:3, :3] @ delta


def _rotate_local(pose: torch.Tensor, r_local: torch.Tensor) -> None:
    pose[:3, :3] = _orthonormalize(pose[:3, :3] @ r_local)


def _make_camera(
    get_projection_matrix,
    pose: torch.Tensor,
    device: torch.device,
    width: int,
    height: int,
    fx_norm: float,
    fy_norm: float,
    focal_scale: float,
    znear: float,
    zfar: float,
):
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
        image.detach()
        .clamp(0.0, 1.0)
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


def _save_state(
    output_dir: Path,
    stem: str,
    image_bgr: np.ndarray,
    pose: torch.Tensor,
    args,
    focal_scale: float,
) -> None:
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
        "  Enter : save FINAL PNG + pose JSON\n"
        "  h     : print this help\n"
        "  Esc   : exit OpenCV UI\n"
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


def _apply_key(
    key: str,
    pose: torch.Tensor,
    start_pose: torch.Tensor,
    focal_scale: float,
    translation_step: float,
    rotation_step: float,
    args,
):
    rerender = True
    if key == "0":
        pose = start_pose.clone()
        focal_scale = 1.0
    elif key == "1":
        translation_step *= 0.5
        rerender = False
    elif key == "2":
        translation_step *= 2.0
        rerender = False
    elif key == "3":
        rotation_step *= 0.5
        rerender = False
    elif key == "4":
        rotation_step *= 2.0
        rerender = False
    elif key == "[":
        focal_scale /= args.fov_step
    elif key == "]":
        focal_scale *= args.fov_step
    elif key == "w":
        _translate_local(pose, 0.0, 0.0, translation_step)
    elif key == "s":
        _translate_local(pose, 0.0, 0.0, -translation_step)
    elif key == "a":
        _translate_local(pose, -translation_step, 0.0, 0.0)
    elif key == "d":
        _translate_local(pose, translation_step, 0.0, 0.0)
    elif key == "r":
        _translate_local(pose, 0.0, -translation_step, 0.0)
    elif key == "f":
        _translate_local(pose, 0.0, translation_step, 0.0)
    elif key == "j":
        _rotate_local(pose, _rot_y(-rotation_step))
    elif key == "l":
        _rotate_local(pose, _rot_y(rotation_step))
    elif key == "i":
        _rotate_local(pose, _rot_x(rotation_step))
    elif key == "k":
        _rotate_local(pose, _rot_x(-rotation_step))
    elif key == "u":
        _rotate_local(pose, _rot_z(-rotation_step))
    elif key == "o":
        _rotate_local(pose, _rot_z(rotation_step))
    else:
        rerender = False
    return pose, focal_scale, translation_step, rotation_step, rerender


def _run_opencv_ui(
    render_current,
    pose: torch.Tensor,
    start_pose: torch.Tensor,
    focal_scale: float,
    translation_step: float,
    rotation_step: float,
    output_dir: Path,
    args,
) -> None:
    candidate_idx = 0
    window = "P001 teaser pose tuner"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    image_bgr = render_current(pose, focal_scale)

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
        raw = cv2.waitKey(0) & 0xFF
        if raw == 27:
            print("[pose-tuner] exit without saving FINAL")
            break
        if raw in (10, 13):
            _save_state(output_dir, "final", image_bgr, pose, args, focal_scale)
            print(f"[pose-tuner] FINAL saved under: {output_dir}")
            break

        key = chr(raw) if 0 <= raw < 128 else ""
        if key == "h":
            _print_help()
            continue
        if key == "p":
            candidate_idx += 1
            stem = f"candidate_{candidate_idx:03d}"
            _save_state(output_dir, stem, image_bgr, pose, args, focal_scale)
            print(f"[pose-tuner] saved {stem}")
            continue

        pose, focal_scale, translation_step, rotation_step, rerender = _apply_key(
            key,
            pose,
            start_pose,
            focal_scale,
            translation_step,
            rotation_step,
            args,
        )
        if rerender:
            image_bgr = render_current(pose, focal_scale)

    cv2.destroyAllWindows()


def _web_html() -> str:
    return r'''<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>P001 teaser pose tuner</title>
<style>
body{font-family:system-ui,sans-serif;margin:18px;background:#111;color:#eee}main{max-width:1100px;margin:auto}img{width:100%;height:auto;background:#000;border:1px solid #444}.row{display:flex;flex-wrap:wrap;gap:8px;margin:10px 0}.group{border:1px solid #444;padding:10px;border-radius:8px}button{font-size:15px;min-width:58px;min-height:38px;background:#252525;color:#eee;border:1px solid #666;border-radius:6px;cursor:pointer}button:hover{background:#333}.status{font-family:ui-monospace,monospace;white-space:pre-wrap;margin:8px 0;color:#ccc}.hint{color:#aaa;font-size:14px}kbd{background:#333;border:1px solid #666;border-radius:3px;padding:1px 5px}
</style>
</head>
<body><main>
<img id="view" alt="Rendered P001 Gaussian map">
<div id="status" class="status">Loading...</div>
<div class="row">
  <div class="group"><b>Move</b><div class="row"><button data-k="w">W Forward</button><button data-k="s">S Back</button><button data-k="a">A Left</button><button data-k="d">D Right</button><button data-k="r">R Up</button><button data-k="f">F Down</button></div></div>
  <div class="group"><b>Rotate</b><div class="row"><button data-k="j">J Yaw-</button><button data-k="l">L Yaw+</button><button data-k="i">I Pitch+</button><button data-k="k">K Pitch-</button><button data-k="u">U Roll-</button><button data-k="o">O Roll+</button></div></div>
</div>
<div class="row">
  <button data-k="[">[ Wider FOV</button><button data-k="]">] Narrower FOV</button>
  <button data-k="1">1 Move /2</button><button data-k="2">2 Move ×2</button>
  <button data-k="3">3 Rot /2</button><button data-k="4">4 Rot ×2</button>
  <button data-k="0">0 Reset</button>
  <button id="candidate">P Save candidate</button><button id="final">Enter Save final</button>
</div>
<p class="hint">Keyboard works when this page has focus: <kbd>WASD</kbd>, <kbd>R/F</kbd>, <kbd>IJKL</kbd>, <kbd>U/O</kbd>, <kbd>[ ]</kbd>, <kbd>1-4</kbd>, <kbd>0</kbd>, <kbd>P</kbd>, <kbd>Enter</kbd>.</p>
<script>
const img=document.getElementById('view'), statusEl=document.getElementById('status');
let busy=false, seq=0;
async function refresh(){
  const s=await (await fetch('/state?'+Date.now(),{cache:'no-store'})).json();
  statusEl.textContent=`move=${s.translation_step_m.toFixed(5)} m   rot=${s.rotation_step_deg.toFixed(4)} deg   focal=${s.focal_scale.toFixed(5)}\nposition=[${s.position.map(v=>v.toFixed(4)).join(', ')}]   candidates=${s.candidate_count}\n${s.message||''}`;
  img.src='/frame.png?v='+(++seq);
}
async function action(key){if(busy)return;busy=true;try{await fetch('/action',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({key})});await refresh();}finally{busy=false;}}
document.querySelectorAll('button[data-k]').forEach(b=>b.addEventListener('click',()=>action(b.dataset.k)));
document.getElementById('candidate').addEventListener('click',()=>action('p'));
document.getElementById('final').addEventListener('click',()=>action('enter'));
window.addEventListener('keydown',e=>{
  if(e.repeat||e.ctrlKey||e.metaKey||e.altKey)return;
  let k=e.key;
  if(k==='Enter') k='enter';
  if(k.length===1) k=k.toLowerCase();
  if(['w','s','a','d','r','f','j','l','i','k','u','o','[',']','1','2','3','4','0','p','enter'].includes(k)){e.preventDefault();action(k);}
});
refresh();
</script>
</main></body></html>'''


def _run_web_ui(
    render_current,
    pose: torch.Tensor,
    start_pose: torch.Tensor,
    focal_scale: float,
    translation_step: float,
    rotation_step: float,
    output_dir: Path,
    args,
) -> None:
    state = {
        "pose": pose,
        "focal_scale": focal_scale,
        "translation_step": translation_step,
        "rotation_step": rotation_step,
        "candidate_idx": 0,
        "image_bgr": None,
        "png": b"",
        "message": "",
    }

    def rerender() -> None:
        state["image_bgr"] = render_current(state["pose"], state["focal_scale"])
        ok, encoded = cv2.imencode(".png", state["image_bgr"])
        if not ok:
            raise RuntimeError("cv2.imencode(.png) failed")
        state["png"] = encoded.tobytes()

    rerender()

    def state_json() -> bytes:
        p = state["pose"]
        payload = {
            "translation_step_m": float(state["translation_step"]),
            "rotation_step_deg": math.degrees(float(state["rotation_step"])),
            "focal_scale": float(state["focal_scale"]),
            "position": [float(x) for x in p[:3, 3].tolist()],
            "candidate_count": int(state["candidate_idx"]),
            "message": str(state["message"]),
        }
        return json.dumps(payload).encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *values):
            return

        def _send(self, status: int, content_type: str, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/":
                self._send(200, "text/html; charset=utf-8", _web_html().encode("utf-8"))
            elif path == "/frame.png":
                self._send(200, "image/png", state["png"])
            elif path == "/state":
                self._send(200, "application/json", state_json())
            else:
                self._send(404, "text/plain; charset=utf-8", b"not found")

        def do_POST(self):
            if urlparse(self.path).path != "/action":
                self._send(404, "text/plain; charset=utf-8", b"not found")
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                request = json.loads(self.rfile.read(length).decode("utf-8"))
                key = str(request.get("key", ""))

                if key == "p":
                    state["candidate_idx"] += 1
                    stem = f"candidate_{state['candidate_idx']:03d}"
                    _save_state(
                        output_dir,
                        stem,
                        state["image_bgr"],
                        state["pose"],
                        args,
                        state["focal_scale"],
                    )
                    state["message"] = f"saved {stem}"
                elif key == "enter":
                    _save_state(
                        output_dir,
                        "final",
                        state["image_bgr"],
                        state["pose"],
                        args,
                        state["focal_scale"],
                    )
                    state["message"] = f"FINAL saved under {output_dir}"
                else:
                    (
                        state["pose"],
                        state["focal_scale"],
                        state["translation_step"],
                        state["rotation_step"],
                        need_render,
                    ) = _apply_key(
                        key,
                        state["pose"],
                        start_pose,
                        state["focal_scale"],
                        state["translation_step"],
                        state["rotation_step"],
                        args,
                    )
                    if need_render:
                        rerender()
                    state["message"] = ""
                self._send(200, "application/json", state_json())
            except Exception as exc:
                body = json.dumps({"error": str(exc)}).encode("utf-8")
                self._send(500, "application/json", body)

    server = HTTPServer((args.host, args.port), Handler)
    print("\n[pose-tuner] headless web UI ready")
    print(f"[pose-tuner] remote URL : http://{args.host}:{args.port}")
    print(
        f"[pose-tuner] VS Code SSH: open the Ports panel, forward port {args.port}, "
        "then click Open in Browser."
    )
    print("[pose-tuner] Ctrl+C stops the server; saved candidates/final remain on disk.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[pose-tuner] web UI stopped")
    finally:
        server.server_close()


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
    parser.add_argument("--ui", choices=("auto", "opencv", "web"), default="auto")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
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

    def render_current(current_pose: torch.Tensor, current_focal_scale: float) -> np.ndarray:
        camera = _make_camera(
            get_projection_matrix,
            current_pose,
            device,
            args.width,
            args.height,
            args.fx_norm,
            args.fy_norm,
            current_focal_scale,
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

    ui = args.ui
    if ui == "auto":
        ui = "opencv" if os.environ.get("DISPLAY") else "web"

    if ui == "web":
        _run_web_ui(
            render_current,
            pose,
            start_pose,
            focal_scale,
            translation_step,
            rotation_step,
            output_dir,
            args,
        )
        return

    try:
        _run_opencv_ui(
            render_current,
            pose,
            start_pose,
            focal_scale,
            translation_step,
            rotation_step,
            output_dir,
            args,
        )
    except cv2.error as exc:
        if args.ui != "auto":
            raise
        print(f"[pose-tuner] OpenCV GUI unavailable: {exc}")
        print("[pose-tuner] falling back to headless web UI")
        _run_web_ui(
            render_current,
            pose,
            start_pose,
            focal_scale,
            translation_step,
            rotation_step,
            output_dir,
            args,
        )


if __name__ == "__main__":
    main()

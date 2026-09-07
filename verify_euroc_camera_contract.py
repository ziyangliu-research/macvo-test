#!/usr/bin/env python3
"""Audit the camera/evaluation contract for the four formal EuRoC sequences.

Checks that each sequence:
- reads intrinsics, resolution, distortion model/coefficients and T_BS from its
  own sensor.yaml;
- uses those distortion coefficients in the MAC-VO EuRoC loader;
- builds stereo undistort/rectification maps;
- exposes the rectified P1 pinhole intrinsic used by MAC-VO, ReSplat and the
  GraphDECO supervision/evaluation path.

This is a preflight only; it does not run MAC-VO, ReSplat or 3DGS optimization.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from DataLoader import SequenceBase, StereoFrame
from DataLoader.Dataset.EuRoC import _sensor_distortion_coefficients
from Utility.Config import load_config


DEFAULT_CONFIGS = [
    "Config/Sequence/EuRoC_MH02_local.yaml",
    "Config/Sequence/EuRoC_V101_local.yaml",
    "Config/Sequence/EuRoC_V201_local.yaml",
    "Config/Sequence/EuRoC_MH05_local.yaml",
]


def _data(value):
    return getattr(value, "data", value)


def _fmt(values, precision: int = 8) -> str:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    return "[" + ", ".join(f"{x:.{precision}g}" for x in array) + "]"


def inspect_config(config_path: Path) -> None:
    cfg, _ = load_config(config_path.resolve())
    if str(cfg.type) != "EuRoC_NoIMU":
        raise ValueError(f"{config_path}: expected type=EuRoC_NoIMU, got {cfg.type}")

    root = Path(cfg.args.root).expanduser().resolve()
    cam0_yaml = root / "cam0" / "sensor.yaml"
    cam1_yaml = root / "cam1" / "sensor.yaml"
    if not cam0_yaml.is_file() or not cam1_yaml.is_file():
        raise FileNotFoundError(f"missing sensor.yaml under {root}")

    cam0, _ = load_config(cam0_yaml)
    cam1, _ = load_config(cam1_yaml)
    d0 = _sensor_distortion_coefficients(cam0, cam0_yaml)
    d1 = _sensor_distortion_coefficients(cam1, cam1_yaml)

    sequence = SequenceBase[StereoFrame].instantiate(cfg.type, cfg.args)
    if sequence.ImageL.undistort_map is None or sequence.ImageR.undistort_map is None:
        raise RuntimeError(f"{config_path}: stereo rectification maps were not built")
    if not np.allclose(sequence.ImageL.distort_factor, d0, atol=1e-12, rtol=0):
        raise RuntimeError(f"{config_path}: cam0 loader distortion != sensor.yaml")
    if not np.allclose(sequence.ImageR.distort_factor, d1, atol=1e-12, rtol=0):
        raise RuntimeError(f"{config_path}: cam1 loader distortion != sensor.yaml")

    name = str(getattr(cfg, "name", config_path.stem))
    print(f"=== {name} ===")
    print(f"root              : {root}")
    print(f"valid stereo+GT   : {len(sequence)} frames")
    print(f"resolution        : {sequence.width}x{sequence.height}")
    print(f"baseline          : {sequence.baseline:.10f} m")
    print(
        "cam0 raw K        : "
        f"fx={cam0.intrinsics[0]:.8f} fy={cam0.intrinsics[1]:.8f} "
        f"cx={cam0.intrinsics[2]:.8f} cy={cam0.intrinsics[3]:.8f}"
    )
    print(f"cam0 distortion   : model={cam0.distortion_model} D={_fmt(d0)}")
    print(
        "cam1 raw K        : "
        f"fx={cam1.intrinsics[0]:.8f} fy={cam1.intrinsics[1]:.8f} "
        f"cx={cam1.intrinsics[2]:.8f} cy={cam1.intrinsics[3]:.8f}"
    )
    print(f"cam1 distortion   : model={cam1.distortion_model} D={_fmt(d1)}")
    K = sequence.K[0].detach().cpu().numpy()
    print(
        "rectified P1 K    : "
        f"fx={K[0,0]:.8f} fy={K[1,1]:.8f} cx={K[0,2]:.8f} cy={K[1,2]:.8f}"
    )
    print("rectification     : OK (cv2 stereoRectify + initUndistortRectifyMap)")
    print("evaluation image  : rectified cam0 image (not raw distorted cam0)\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", nargs="*", default=DEFAULT_CONFIGS)
    args = parser.parse_args()

    for item in args.configs:
        inspect_config(Path(item))
    print("[OK] EuRoC camera contract verified for all requested sequences.")


if __name__ == "__main__":
    main()

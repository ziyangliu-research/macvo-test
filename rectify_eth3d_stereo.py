#!/usr/bin/env python3
"""Rectify ETH3D RGB stereo with camera-2 as the left/reference camera.

ETH3D processed SLAM data already contains undistorted pinhole images, but the
second RGB camera is not stereo-rectified with the first.  This utility uses the
dataset calibration to build a conventional horizontal stereo pair:

    Left  = camera 2 / rgb2
    Right = camera 1 / rgb

`extrinsics_1_2.txt` maps points from camera 2 to camera 1, which is exactly the
R,T convention OpenCV stereoRectify expects for this left/right ordering.

Important GT convention:
ETH3D groundtruth.txt is the camera-1/original-right c2w pose.  After converting
it to camera 2, the pose must ALSO be rotated into the rectified-left camera
frame using R1.  The output groundtruth_left.txt therefore corresponds to the
actual rectified image_left camera used by MAC-VO/ReSplat.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np


def read_intrinsics(path: Path) -> np.ndarray:
    vals = np.loadtxt(path).reshape(-1)
    if len(vals) != 4:
        raise ValueError(f"Expected fx fy cx cy in {path}, got {vals}")
    fx, fy, cx, cy = [float(v) for v in vals]
    return np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64
    )


def read_extrinsic(path: Path) -> np.ndarray:
    M = np.loadtxt(path).reshape(3, 4)
    T = np.eye(4, dtype=np.float64)
    T[:3, :] = M
    return T


def quat_xyzw_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm <= 0:
        raise ValueError("zero quaternion")
    q /= norm
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rot_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    m00, m01, m02 = R[0]
    m10, m11, m12 = R[1]
    m20, m21, m22 = R[2]
    tr = m00 + m11 + m22
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (m21 - m12) / S
        qy = (m02 - m20) / S
        qz = (m10 - m01) / S
    elif m00 > m11 and m00 > m22:
        S = np.sqrt(1.0 + m00 - m11 - m22) * 2
        qw = (m21 - m12) / S
        qx = 0.25 * S
        qy = (m01 + m10) / S
        qz = (m02 + m20) / S
    elif m11 > m22:
        S = np.sqrt(1.0 + m11 - m00 - m22) * 2
        qw = (m02 - m20) / S
        qx = (m01 + m10) / S
        qy = 0.25 * S
        qz = (m12 + m21) / S
    else:
        S = np.sqrt(1.0 + m22 - m00 - m11) * 2
        qw = (m10 - m01) / S
        qx = (m02 + m20) / S
        qy = (m12 + m21) / S
        qz = 0.25 * S
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    q /= np.linalg.norm(q)
    # Canonical sign only for stable text output.
    if q[3] < 0:
        q = -q
    return q


def pose_line_to_matrix(vals: np.ndarray) -> tuple[float, np.ndarray]:
    if vals.size != 8:
        raise ValueError(f"expected ETH3D timestamp+pose row with 8 values, got {vals}")
    timestamp = float(vals[0])
    tx, ty, tz = vals[1:4]
    qx, qy, qz, qw = vals[4:8]
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quat_xyzw_to_rot(qx, qy, qz, qw)
    T[:3, 3] = [tx, ty, tz]
    return timestamp, T


def matrix_to_pose_line(timestamp: float, T: np.ndarray) -> str:
    qx, qy, qz, qw = rot_to_quat_xyzw(T[:3, :3])
    tx, ty, tz = T[:3, 3]
    return (
        f"{timestamp:.9f} "
        f"{tx:.9f} {ty:.9f} {tz:.9f} "
        f"{qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}"
    )


def pad_to_size(img: np.ndarray, width: int, height: int) -> np.ndarray:
    h, w = img.shape[:2]
    if w > width or h > height:
        raise ValueError(f"Cannot pad image of size {w}x{h} to {width}x{height}")
    # Pad only right/bottom so the original pixel origin and K stay unchanged.
    return cv2.copyMakeBorder(
        img,
        0,
        height - h,
        0,
        width - w,
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )


def transform4(R: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    return T


def camera_fov_deg(K: np.ndarray, width: int, height: int) -> dict[str, float]:
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    hfov = np.degrees(np.arctan2(cx, fx) + np.arctan2((width - 1) - cx, fx))
    vfov = np.degrees(np.arctan2(cy, fy) + np.arctan2((height - 1) - cy, fy))
    return {"horizontal": float(hfov), "vertical": float(vfov)}


def valid_map_fraction(mapx: np.ndarray, mapy: np.ndarray, width: int, height: int) -> float:
    valid = (mapx >= 0.0) & (mapx <= width - 1.0) & (mapy >= 0.0) & (mapy <= height - 1.0)
    return float(valid.mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.0,
        help="OpenCV stereoRectify alpha. 0 crops invalid borders; 1 preserves all source pixels.",
    )
    args = parser.parse_args()

    src = Path(args.input).expanduser().resolve()
    dst = Path(args.output).expanduser().resolve()

    left_dir = src / "rgb2"   # ETH3D camera 2 = physical left
    right_dir = src / "rgb"   # ETH3D camera 1 = physical right
    K_left = read_intrinsics(src / "calibration2.txt")
    K_right = read_intrinsics(src / "calibration.txt")

    # ETH3D: extrinsics_1_2 maps camera-2 points -> camera-1 points.
    # With our ordering left=cam2, right=cam1 this is T_right_from_left.
    T_right_from_left = read_extrinsic(src / "extrinsics_1_2.txt")
    R_right_from_left = T_right_from_left[:3, :3]
    t_right_from_left = T_right_from_left[:3, 3]

    if not np.allclose(R_right_from_left.T @ R_right_from_left, np.eye(3), atol=1e-5):
        raise ValueError("extrinsics_1_2 rotation is not orthonormal")
    if np.linalg.det(R_right_from_left) < 0.999 or np.linalg.det(R_right_from_left) > 1.001:
        raise ValueError("extrinsics_1_2 rotation determinant is not +1")

    left_files = {p.name: p for p in left_dir.glob("*.png")}
    right_files = {p.name: p for p in right_dir.glob("*.png")}
    common_names = sorted(set(left_files) & set(right_files))
    if not common_names:
        raise RuntimeError("No synchronized stereo image pairs found")
    print(f"Found {len(common_names)} synchronized pairs")

    first_left = cv2.imread(str(left_files[common_names[0]]), cv2.IMREAD_COLOR)
    first_right = cv2.imread(str(right_files[common_names[0]]), cv2.IMREAD_COLOR)
    if first_left is None or first_right is None:
        raise RuntimeError("Failed to read first stereo pair")
    h_l, w_l = first_left.shape[:2]
    h_r, w_r = first_right.shape[:2]
    common_w, common_h = max(w_l, w_r), max(h_l, h_r)
    image_size = (common_w, common_h)

    print(f"Original LEFT : {w_l} x {h_l}")
    print(f"Original RIGHT: {w_r} x {h_r}")
    print(f"Common canvas : {common_w} x {common_h}")

    # Processed ETH3D images are already undistorted pinhole images.
    D_left = np.zeros(5, dtype=np.float64)
    D_right = np.zeros(5, dtype=np.float64)

    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K_left,
        D_left,
        K_right,
        D_right,
        image_size,
        R_right_from_left,
        t_right_from_left,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=args.alpha,
        newImageSize=image_size,
    )

    map1x, map1y = cv2.initUndistortRectifyMap(
        K_left, D_left, R1, P1[:, :3], image_size, cv2.CV_32FC1
    )
    map2x, map2y = cv2.initUndistortRectifyMap(
        K_right, D_right, R2, P2[:, :3], image_size, cv2.CV_32FC1
    )

    K_rect_left = P1[:, :3].copy()
    K_rect_right = P2[:, :3].copy()
    if not np.allclose(K_rect_left, K_rect_right, atol=1e-6, rtol=1e-7):
        raise RuntimeError(
            "CALIB_ZERO_DISPARITY did not produce matching rectified K matrices:\n"
            f"P1 K=\n{K_rect_left}\nP2 K=\n{K_rect_right}"
        )

    # For horizontal rectification P2[0,3] = -fx * B when the physical right
    # camera center is +B on the rectified-left x axis.
    if abs(float(P2[1, 3])) > 1e-6 * max(1.0, abs(float(P2[0, 3]))):
        raise RuntimeError(f"unexpected vertical rectified baseline in P2: {P2[:, 3]}")
    signed_baseline_rect = -float(P2[0, 3]) / float(P2[0, 0])
    baseline_original = float(np.linalg.norm(t_right_from_left))
    baseline_rectified = abs(signed_baseline_rect)
    if abs(baseline_rectified - baseline_original) > 1e-4:
        raise RuntimeError(
            f"rectified baseline mismatch: original={baseline_original:.9f} "
            f"from_P2={baseline_rectified:.9f}"
        )

    out_left = dst / "image_left"
    out_right = dst / "image_right"
    out_left.mkdir(parents=True, exist_ok=True)
    out_right.mkdir(parents=True, exist_ok=True)

    timestamps: list[str] = []
    for idx, name in enumerate(common_names):
        left = cv2.imread(str(left_files[name]), cv2.IMREAD_COLOR)
        right = cv2.imread(str(right_files[name]), cv2.IMREAD_COLOR)
        if left is None or right is None:
            raise RuntimeError(f"Failed to read pair {name}")
        left = pad_to_size(left, common_w, common_h)
        right = pad_to_size(right, common_w, common_h)
        left_rect = cv2.remap(
            left, map1x, map1y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
        )
        right_rect = cv2.remap(
            right, map2x, map2y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
        )
        cv2.imwrite(str(out_left / name), left_rect)
        cv2.imwrite(str(out_right / name), right_rect)
        timestamps.append(Path(name).stem)
        if idx % 50 == 0 or idx == len(common_names) - 1:
            print(f"[{idx + 1:4d}/{len(common_names)}] {name}")

    (dst / "timestamps.txt").write_text("\n".join(timestamps) + "\n", encoding="utf-8")

    # R1 maps original-left camera coordinates -> rectified-left coordinates.
    # Therefore rectified-left c2w = original-left c2w @ inv(R1).
    T_rect_from_left_original = transform4(R1)
    T_left_original_from_rect = np.linalg.inv(T_rect_from_left_original)

    calibration = {
        "source": str(src),
        "camera_convention": {
            "left": "ETH3D camera 2 / rgb2",
            "right": "ETH3D camera 1 / rgb",
            "extrinsics_1_2": "T_right_original_from_left_original",
            "groundtruth_left": "T_world_from_left_rectified",
        },
        "original_left_size": [w_l, h_l],
        "original_right_size": [w_r, h_r],
        "rectified_size": [common_w, common_h],
        "baseline_original_m": baseline_original,
        "baseline_rectified_m": baseline_rectified,
        "signed_baseline_rectified_m": signed_baseline_rect,
        "K_left_original": K_left.tolist(),
        "K_right_original": K_right.tolist(),
        "T_right_original_from_left_original": T_right_from_left.tolist(),
        "R1": R1.tolist(),
        "R2": R2.tolist(),
        "P1": P1.tolist(),
        "P2": P2.tolist(),
        "Q": Q.tolist(),
        "K_rectified_left": K_rect_left.tolist(),
        "K_rectified_right": K_rect_right.tolist(),
        "T_rectified_left_from_original_left": T_rect_from_left_original.tolist(),
        "T_original_left_from_rectified_left": T_left_original_from_rect.tolist(),
        "roi_left": list(map(int, roi1)),
        "roi_right": list(map(int, roi2)),
        "alpha": float(args.alpha),
        "rectified_fov_deg": camera_fov_deg(K_rect_left, common_w, common_h),
        "rectification_map_valid_fraction_left": valid_map_fraction(
            map1x, map1y, common_w, common_h
        ),
        "rectification_map_valid_fraction_right": valid_map_fraction(
            map2x, map2y, common_w, common_h
        ),
    }
    (dst / "calibration.json").write_text(
        json.dumps(calibration, indent=2), encoding="utf-8"
    )

    gt_in = src / "groundtruth.txt"
    if gt_in.exists():
        gt_rectified_out = dst / "groundtruth_left.txt"
        gt_left_original_out = dst / "groundtruth_left_original_camera2.txt"
        with gt_in.open("r", encoding="utf-8") as fi, \
             gt_rectified_out.open("w", encoding="utf-8") as fo_rect, \
             gt_left_original_out.open("w", encoding="utf-8") as fo_orig:
            for line in fi:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                vals = np.fromstring(line, sep=" ")
                if len(vals) != 8:
                    continue
                timestamp, T_world_right_original = pose_line_to_matrix(vals)

                # extrinsics_1_2 = T_right_original_from_left_original
                T_world_left_original = T_world_right_original @ T_right_from_left
                T_world_left_rectified = (
                    T_world_left_original @ T_left_original_from_rect
                )

                fo_orig.write(matrix_to_pose_line(timestamp, T_world_left_original) + "\n")
                fo_rect.write(matrix_to_pose_line(timestamp, T_world_left_rectified) + "\n")

        shutil.copy2(gt_in, dst / "groundtruth_right_original.txt")

    print("\n===================================")
    print("Rectification complete")
    print("===================================")
    print(f"Output             : {dst}")
    print(f"Pairs              : {len(common_names)}")
    print(f"Rectified size     : {common_w} x {common_h}")
    print(f"Baseline original  : {baseline_original:.6f} m")
    print(f"Baseline from P2   : {baseline_rectified:.6f} m")
    print(f"Signed baseline    : {signed_baseline_rect:.6f} m")
    fov = calibration["rectified_fov_deg"]
    print(f"Rectified FoV      : H={fov['horizontal']:.2f} deg V={fov['vertical']:.2f} deg")
    print(
        "Valid map fraction : "
        f"L={calibration['rectification_map_valid_fraction_left']:.4f} "
        f"R={calibration['rectification_map_valid_fraction_right']:.4f}"
    )
    print("\nP1:")
    print(P1)
    print("\nP2:")
    print(P2)


if __name__ == "__main__":
    main()

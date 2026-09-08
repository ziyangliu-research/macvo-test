#!/usr/bin/env python3
"""Runtime support for rectified ETH3D RGB stereo.

The dataset is preprocessed by rectify_eth3d_stereo.py.  The resulting image
pairs are already horizontal stereo with a shared pinhole K.  ReSplat therefore
uses the exact per-frame rectified K supplied by the loader and a full-FoV resize
rather than the TartanAir-specific static-K path.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


def _resize_full_fov_image_and_intrinsic(
    image: Image.Image,
    K_pixel: torch.Tensor,
    image_shape: tuple[int, int],
    normalize_intrinsics: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    import torchvision.transforms.functional as TF

    image = image.convert("RGB")
    original_w, original_h = image.size
    target_h, target_w = image_shape
    if min(original_w, original_h, target_w, target_h) <= 0:
        raise ValueError(
            f"invalid ETH3D resize {original_w}x{original_h} -> {target_w}x{target_h}"
        )
    scale_x = target_w / original_w
    scale_y = target_h / original_h
    image = image.resize((target_w, target_h), Image.Resampling.BILINEAR)
    tensor = TF.to_tensor(image)

    K = K_pixel.clone().float()
    K[0, :] *= scale_x
    K[1, :] *= scale_y
    K[2, :] = torch.tensor([0.0, 0.0, 1.0], dtype=K.dtype)
    if normalize_intrinsics:
        K[0, :] /= target_w
        K[1, :] /= target_h
        K[2, :] = torch.tensor([0.0, 0.0, 1.0], dtype=K.dtype)
    return tensor, K


def install_eth3d_runtime_support() -> None:
    """Install ETH3D-only descriptor and ReSplat dynamic-K/full-FoV patches."""
    from Utility.Config import load_config
    from async_pipeline.contracts import FrameDescriptor
    from async_pipeline.macvo_runtime import MacvoPoseFrontend
    from async_pipeline.resplat_runtime import ResplatPacketGenerator
    import torchvision.transforms.functional as TF

    original_descriptors = MacvoPoseFrontend.descriptors
    if not getattr(original_descriptors, "_eth3d_rectified_patch", False):
        def descriptors_eth3d(self):
            self.initialize()
            data_cfg, _ = load_config(self.config.data_config.expanduser().resolve())
            if str(data_cfg.type) != "ETH3D_Rectified":
                return original_descriptors(self)

            root = Path(data_cfg.args.root).expanduser().resolve()
            left_files = sorted((root / "image_left").glob("*.png"))
            right_files = sorted((root / "image_right").glob("*.png"))
            timestamps = [
                line.strip()
                for line in (root / "timestamps.txt").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if len(left_files) != len(right_files) or len(left_files) != len(timestamps):
                raise ValueError("ETH3D descriptor source counts are inconsistent")

            descriptors = []
            for sequence_index, source_index in enumerate(
                range(self.config.start_index, self.config.end_index)
            ):
                if source_index >= len(left_files):
                    break
                descriptors.append(
                    FrameDescriptor(
                        sequence_index=sequence_index,
                        frame_index=source_index,
                        timestamp_ns=int(round(float(timestamps[source_index]) * 1e9)),
                        left_path=left_files[source_index],
                        right_path=right_files[source_index],
                        is_test=False,
                    )
                )
            return descriptors

        descriptors_eth3d._eth3d_rectified_patch = True  # type: ignore[attr-defined]
        MacvoPoseFrontend.descriptors = descriptors_eth3d

    original_load_inputs = ResplatPacketGenerator._load_input_images
    if not getattr(original_load_inputs, "_eth3d_dynamic_k_full_fov_patch", False):
        def load_inputs_eth3d(self, frame_input):
            if self.config.input_mode != "shared_tensors":
                return original_load_inputs(self, frame_input)

            frame_input.validate(deep=self.config.strict_validation)
            K_pixel = frame_input.intrinsic_pixel.detach().cpu().float().contiguous()
            left_pil = TF.to_pil_image(frame_input.left_image.detach().cpu().clamp(0, 1))
            right_pil = TF.to_pil_image(frame_input.right_image.detach().cpu().clamp(0, 1))
            left, K_left = _resize_full_fov_image_and_intrinsic(
                left_pil, K_pixel, self.image_shape, self.normalize_intrinsics
            )
            right, K_right = _resize_full_fov_image_and_intrinsic(
                right_pil, K_pixel, self.image_shape, self.normalize_intrinsics
            )
            return left, right, K_left, K_right

        load_inputs_eth3d._eth3d_dynamic_k_full_fov_patch = True  # type: ignore[attr-defined]
        ResplatPacketGenerator._load_input_images = load_inputs_eth3d


def evaluate_pose_eth3d(
    runner,
    resolved: dict[str, Any],
    num_frames: int,
    output: Path,
) -> dict[str, Any]:
    """Evaluate MAC-VO against rectified-left ETH3D GT with SE(3)/Sim(3)."""
    from DataLoader import SequenceBase, StereoFrame
    from Utility.Config import load_config
    from run_async_pipeline_metrics import (
        collect_predicted_trajectory,
        save_tum,
        trajectory_metric,
    )

    predicted_all, valid_mask = collect_predicted_trajectory(runner, num_frames)
    data_cfg, _ = load_config(Path(resolved["paths"]["data_config"]))
    if str(data_cfg.type) != "ETH3D_Rectified":
        raise ValueError(f"expected ETH3D_Rectified, got {data_cfg.type}")
    sequence = SequenceBase[StereoFrame].instantiate(data_cfg.type, data_cfg.args)
    if not hasattr(sequence, "gt_pose_data") or sequence.gt_pose_data is None:
        raise RuntimeError("ETH3D rectified sequence has no interpolated GT")

    start = int(resolved["sequence"]["start_index"])
    selected = np.arange(start, start + predicted_all.shape[0], dtype=np.int64)
    if selected.size == 0 or int(selected[-1]) >= len(sequence):
        raise IndexError(
            f"ETH3D GT indices outside sequence: selected={selected[[0,-1]] if selected.size else selected}, "
            f"len={len(sequence)}"
        )

    gt_absolute = sequence.gt_pose_data[selected].matrix().detach().cpu().double().numpy()
    gt_first_inv = np.linalg.inv(gt_absolute[0])
    gt_all = gt_first_inv[None] @ gt_absolute

    predicted = predicted_all[valid_mask]
    ground_truth = gt_all[valid_mask]
    valid_indices = selected[valid_mask]
    if predicted.shape[0] < 3:
        raise RuntimeError("fewer than three valid MAC-VO poses are available")

    report = {
        "coordinate_convention": "metric OpenCV rectified-left c2w, first selected frame identity",
        "gt_source": "ETH3D groundtruth.txt -> camera2(left) -> rectified-left, interpolated to image timestamps",
        "num_requested_poses": int(predicted_all.shape[0]),
        "num_valid_poses": int(predicted.shape[0]),
        "num_skipped_need_interp": int((~valid_mask).sum()),
        "raw": trajectory_metric(predicted, ground_truth, alignment="raw"),
        "se3": trajectory_metric(predicted, ground_truth, alignment="se3"),
        "sim3": trajectory_metric(predicted, ground_truth, alignment="sim3"),
    }

    np.savez_compressed(
        output / "macvo_eth3d_trajectory.npz",
        selected_original_indices=selected,
        valid_mask=valid_mask,
        T_pred_c2w_opencv=predicted_all,
        T_gt_c2w_opencv=gt_all,
    )
    save_tum(output / "macvo_pred_valid.tum", valid_indices, predicted)
    save_tum(output / "macvo_gt_valid.tum", valid_indices, ground_truth)
    (output / "pose_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report

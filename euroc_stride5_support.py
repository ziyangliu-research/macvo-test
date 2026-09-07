#!/usr/bin/env python3
"""Runtime support for the downsampled EuRoC evaluation protocol.

The formal EuRoC protocol used by the experiment launchers is:
  1. load the existing EuRoC stereo sequence (official sensor.yaml calibration,
     stereo synchronization/rectification, and GT timestamp masking);
  2. retain every N-th valid synchronized frame (default N=5);
  3. apply the normal strict held-out split on this retained local sequence.

This module deliberately installs the stride only for ``EuRoC_NoIMU`` runs. It
also makes the shared-tensor ReSplat path consume the per-frame rectified pixel
intrinsics supplied by the EuRoC loader instead of the TartanAir static K.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch


def frame_stride() -> int:
    stride = int(os.environ.get("PIPELINE_FRAME_STRIDE", "5"))
    if stride <= 0:
        raise ValueError("PIPELINE_FRAME_STRIDE must be positive")
    return stride


def install_euroc_runtime_support() -> None:
    """Install EuRoC-only stride, pose-coordinate, and ReSplat-K patches."""
    from DataLoader.SequenceBase import SequenceBase
    from Utility.Config import load_config
    from async_pipeline.macvo_runtime import (
        MacvoPoseFrontend,
        _pose7_xyzw_to_matrix,
        _tartan_from_cv,
    )
    from async_pipeline.resplat_runtime import (
        ResplatPacketGenerator,
        process_pil_image_and_intrinsic,
    )
    import torchvision.transforms.functional as TF

    # ------------------------------------------------------------------
    # 1) Temporal downsampling BEFORE the strict 8:2 held-out split.
    # MacvoPoseFrontend.initialize() calls SequenceBase.clip(start, end)
    # on the raw sequence before smart_transform(), which is exactly where
    # the temporal stride must be applied.
    # ------------------------------------------------------------------
    original_clip = SequenceBase.clip
    if not getattr(original_clip, "_euroc_stride5_patch", False):
        def clip_with_euroc_stride(self, start_idx=None, end_idx=None, step=None):
            if step is None and self.name() == "EuRoC_NoIMU":
                step = frame_stride()
            return original_clip(self, start_idx, end_idx, step)

        clip_with_euroc_stride._euroc_stride5_patch = True  # type: ignore[attr-defined]
        SequenceBase.clip = clip_with_euroc_stride  # type: ignore[method-assign]

    # descriptors() must contain the same number of entries as the strided
    # sequence. Paths are metadata only for shared_tensors mode; EuRoC image
    # filenames are timestamps rather than integer indices.
    original_descriptors = MacvoPoseFrontend.descriptors
    if not getattr(original_descriptors, "_euroc_stride5_patch", False):
        def descriptors_with_stride(self):
            self.initialize()
            data_cfg, _ = load_config(self.config.data_config.expanduser().resolve())
            if str(data_cfg.type) != "EuRoC_NoIMU":
                return original_descriptors(self)

            from async_pipeline.contracts import FrameDescriptor
            root = Path(data_cfg.args.root).expanduser().resolve()
            stride = frame_stride()
            descriptors = []
            for sequence_index, source_index in enumerate(
                range(self.config.start_index, self.config.end_index, stride)
            ):
                descriptors.append(
                    FrameDescriptor(
                        sequence_index=sequence_index,
                        frame_index=source_index,
                        timestamp_ns=source_index,
                        left_path=root / "cam0" / "data" / f"source_{source_index}.png",
                        right_path=root / "cam1" / "data" / f"source_{source_index}.png",
                        is_test=False,
                    )
                )
            return descriptors

        descriptors_with_stride._euroc_stride5_patch = True  # type: ignore[attr-defined]
        MacvoPoseFrontend.descriptors = descriptors_with_stride

    # ------------------------------------------------------------------
    # 2) MAC-VO graph poses are stereo-sensor poses. For non-identity EuRoC
    # extrinsics, do not apply the old body-frame conjugation used by the
    # TartanAir adapter. Compose only the NED-camera -> OpenCV-camera basis.
    # TartanAir is unaffected because this override is installed only by the
    # EuRoC wrapper runners.
    # ------------------------------------------------------------------
    original_opencv_pose = MacvoPoseFrontend._opencv_relative_pose
    if not getattr(original_opencv_pose, "_euroc_sensor_pose_patch", False):
        def euroc_opencv_relative_pose(self, index: int) -> torch.Tensor:
            graph = self.system.graph
            sensor_pose7 = graph.frames.data["pose"].tensor[index].detach().cpu()
            absolute = _pose7_xyzw_to_matrix(sensor_pose7) @ _tartan_from_cv()
            if self._T0_inv is None:
                first7 = graph.frames.data["pose"].tensor[0].detach().cpu()
                first = _pose7_xyzw_to_matrix(first7) @ _tartan_from_cv()
                self._T0_inv = torch.linalg.inv(first)
            return (self._T0_inv @ absolute).float()

        euroc_opencv_relative_pose._euroc_sensor_pose_patch = True  # type: ignore[attr-defined]
        MacvoPoseFrontend._opencv_relative_pose = euroc_opencv_relative_pose

    # ------------------------------------------------------------------
    # 3) In shared_tensors mode the EuRoC loader already supplies the exact
    # rectified K from sensor.yaml + stereoRectify. Use that dynamic K for
    # ReSplat resize/crop rather than the static TartanAir K in config.
    # ------------------------------------------------------------------
    original_load_inputs = ResplatPacketGenerator._load_input_images
    if not getattr(original_load_inputs, "_euroc_dynamic_k_patch", False):
        def load_inputs_with_dynamic_k(self, frame_input):
            if self.config.input_mode != "shared_tensors":
                return original_load_inputs(self, frame_input)

            frame_input.validate(deep=self.config.strict_validation)
            K_pixel = frame_input.intrinsic_pixel.detach().cpu().float().contiguous()
            left_pil = TF.to_pil_image(
                frame_input.left_image.detach().cpu().clamp(0, 1)
            )
            right_pil = TF.to_pil_image(
                frame_input.right_image.detach().cpu().clamp(0, 1)
            )
            left, K_left = process_pil_image_and_intrinsic(
                left_pil, K_pixel, self.image_shape, self.normalize_intrinsics
            )
            right, K_right = process_pil_image_and_intrinsic(
                right_pil, K_pixel, self.image_shape, self.normalize_intrinsics
            )
            return left, right, K_left, K_right

        load_inputs_with_dynamic_k._euroc_dynamic_k_patch = True  # type: ignore[attr-defined]
        ResplatPacketGenerator._load_input_images = load_inputs_with_dynamic_k


def evaluate_pose_euroc(
    runner,
    resolved: dict[str, Any],
    num_frames: int,
    output: Path,
) -> dict[str, Any]:
    """Evaluate MAC-VO against EuRoC dataset GT after the same temporal stride.

    The existing EuRoC loader already synchronizes stereo timestamps, masks to
    the GT-covered interval, and interpolates body GT to camera timestamps.
    We reuse that exact loader state and convert body GT to the left-camera
    sensor pose with its calibrated T_BS before SE(3)/Sim(3) alignment.
    """
    import pypose as pp
    from DataLoader import SequenceBase, StereoFrame
    from Utility.Config import load_config
    from async_pipeline.macvo_runtime import _tartan_from_cv
    from run_async_pipeline_metrics import (
        collect_predicted_trajectory,
        save_tum,
        trajectory_metric,
    )

    predicted_all, valid_mask = collect_predicted_trajectory(runner, num_frames)

    data_cfg, _ = load_config(Path(resolved["paths"]["data_config"]))
    if str(data_cfg.type) != "EuRoC_NoIMU":
        raise ValueError(
            f"EuRoC evaluator requires type=EuRoC_NoIMU, got {data_cfg.type}"
        )
    sequence = SequenceBase[StereoFrame].instantiate(data_cfg.type, data_cfg.args)
    if not hasattr(sequence, "gt_pose_data") or sequence.gt_pose_data is None:
        raise RuntimeError("EuRoC sequence has no GT poses; set gt_pose: true")
    if not hasattr(sequence, "T_BS_lcam"):
        raise RuntimeError("EuRoC sequence has no calibrated left-camera T_BS")

    start = int(resolved["sequence"]["start_index"])
    end = int(resolved["sequence"]["end_index"])
    stride = frame_stride()
    selected = np.arange(start, min(end, len(sequence)), stride, dtype=np.int64)
    selected = selected[: predicted_all.shape[0]]
    if selected.size != predicted_all.shape[0]:
        raise RuntimeError(
            f"GT/estimate length mismatch: selected_gt={selected.size}, "
            f"predicted={predicted_all.shape[0]}"
        )

    gt_body = pp.SE3(sequence.gt_pose_data[selected])
    T_BS_lcam = pp.SE3(sequence.T_BS_lcam)
    # EuRoC T_BS is body-from-sensor. Therefore W<-cam = W<-body @ body<-cam.
    gt_sensor = gt_body @ T_BS_lcam
    gt_absolute = gt_sensor.matrix().detach().cpu().double().numpy()
    cv_basis = _tartan_from_cv(dtype=torch.float64).numpy()
    gt_absolute = gt_absolute @ cv_basis[None]
    gt_first_inv = np.linalg.inv(gt_absolute[0])
    gt_all = gt_first_inv[None] @ gt_absolute

    predicted = predicted_all[valid_mask]
    ground_truth = gt_all[valid_mask]
    valid_indices = selected[valid_mask]
    if predicted.shape[0] < 3:
        raise RuntimeError("fewer than three valid MAC-VO poses are available")

    report = {
        "coordinate_convention": "metric OpenCV c2w, first retained frame identity",
        "gt_source": "EuRoC state_groundtruth_estimate0 via existing synchronized/interpolated dataset loader",
        "frame_stride_before_split": stride,
        "num_requested_poses": int(predicted_all.shape[0]),
        "num_valid_poses": int(predicted.shape[0]),
        "num_skipped_need_interp": int((~valid_mask).sum()),
        "raw": trajectory_metric(predicted, ground_truth, alignment="raw"),
        "se3": trajectory_metric(predicted, ground_truth, alignment="se3"),
        "sim3": trajectory_metric(predicted, ground_truth, alignment="sim3"),
    }

    np.savez_compressed(
        output / "macvo_async_trajectory.npz",
        selected_original_indices=selected,
        valid_mask=valid_mask,
        T_pred_c2w_opencv=predicted_all,
        T_gt_c2w_opencv=gt_all,
    )
    save_tum(output / "macvo_pred_valid.tum", valid_indices, predicted)
    save_tum(output / "macvo_gt_valid.tum", valid_indices, ground_truth)
    (output / "pose_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report

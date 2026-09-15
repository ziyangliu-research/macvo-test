#!/usr/bin/env python3
"""P001 V2 diagnostic using exact per-frame left/right pose-file stereo extrinsics.

This is an A/B diagnostic against the normal runtime path, which synthesizes the
right camera from a fixed 0.25 m stereo rig.  Here we keep the same images,
intrinsics, checkpoint, ReSplat model, backend, and online policy, but replace
only ReSplat's right-camera context extrinsic with

    T_left_from_right = inv(Twc_left) @ Twc_right

computed from TartanAir V2's pose_lcam_front.txt and pose_rcam_front.txt.

The left context camera remains the local packet origin.  This script exists to
isolate stereo-rig convention/sign/rotation errors; it is not a new paper
protocol.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch

import run_pipeline_execution_benchmark_repro_p001_v2_diagnostic as diag
import run_pipeline_execution_benchmark_repro_p001_teaser as teaser
import run_pipeline_execution_benchmark_repro_empty_safe as safe


def _install_posefile_stereo_patch() -> None:
    from async_pipeline.resplat_runtime import ResplatPacketGenerator

    original_make_batch = ResplatPacketGenerator._make_batch
    if getattr(original_make_batch, "_p001_posefile_stereo_patch", False):
        return

    data_root = Path(
        os.environ.get(
            "PIPELINE_P001_DIAG_DATA_ROOT",
            "/home/shiyo/Desktop/Datasets/tartanair_v2/House/Data_easy/P001",
        )
    ).expanduser().resolve()
    left_path = data_root / "pose_lcam_front.txt"
    right_path = data_root / "pose_rcam_front.txt"
    if not left_path.is_file() or not right_path.is_file():
        raise FileNotFoundError(
            f"pose-file stereo diagnostic requires {left_path} and {right_path}"
        )

    left_rows = diag._load_tartan_pose_rows(left_path)
    right_rows = diag._load_tartan_pose_rows(right_path)
    count = min(len(left_rows), len(right_rows))
    if count <= 0:
        raise RuntimeError("no overlapping P001 left/right pose rows")

    relative_rows = [
        torch.linalg.inv(left_rows[i]) @ right_rows[i]
        for i in range(count)
    ]

    def make_batch_with_posefile_stereo(self, frame_input, T_left_c2w):
        batch = original_make_batch(self, frame_input, T_left_c2w)
        frame_index = int(frame_input.descriptor.frame_index)
        if frame_index < 0 or frame_index >= len(relative_rows):
            raise IndexError(
                f"P001 pose-file stereo frame {frame_index} outside pose rows [0,{len(relative_rows)})"
            )

        relative = relative_rows[frame_index].to(dtype=torch.float32)
        T_left = T_left_c2w.detach().cpu().float()
        T_right = T_left @ relative
        batch["context"]["extrinsics"][0, 0] = T_left
        batch["context"]["extrinsics"][0, 1] = T_right
        # Target remains the left camera, exactly as in the normal runtime.
        batch["target"]["extrinsics"][0, 0] = T_left
        return batch

    make_batch_with_posefile_stereo._p001_posefile_stereo_patch = True  # type: ignore[attr-defined]
    ResplatPacketGenerator._make_batch = make_batch_with_posefile_stereo

    first = relative_rows[0]
    t = first[:3, 3]
    print(
        "[P001 pose-file stereo] enabled: "
        "ReSplat right extrinsic = inv(Twc_left_posefile) @ Twc_right_posefile; "
        f"frame0 translation_cv={t.tolist()} baseline={float(torch.linalg.vector_norm(t)):.8f}m",
        flush=True,
    )


if __name__ == "__main__":
    diag._write_v2_camera_audit()
    _install_posefile_stereo_patch()
    diag._install_resplat_packet_diagnostics()
    diag._install_final_map_diagnostics()
    teaser._install_p001_teaser_render()
    safe.repro.main()

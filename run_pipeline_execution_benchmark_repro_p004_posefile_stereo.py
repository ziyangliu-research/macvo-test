#!/usr/bin/env python3
"""P004 teaser/diagnostic run using exact per-frame stereo pose-file extrinsics.

Compared with the normal P004 teaser runner, this changes only ReSplat's right
context-camera extrinsic.  Instead of synthesizing it from a scalar 0.25 m
baseline, every timestamp uses

    T_left_from_right = inv(Twc_left) @ Twc_right

from TartanAir V2's pose_lcam_front.txt and pose_rcam_front.txt.

The left camera remains the local packet origin.  All ReSplat packet renders,
final-map input-view renders, final PLY, and the wide first-view teaser artifacts
are produced by the normal P004 teaser wrapper.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch

import audit_tartanair_v2_stereo_pose as audit
import run_pipeline_execution_benchmark_repro_p004_teaser as p004
import run_pipeline_execution_benchmark_repro_empty_safe as safe


def _data_root() -> Path:
    return Path(
        os.environ.get(
            "PIPELINE_P004_DATA_ROOT",
            "/home/shiyo/Desktop/Datasets/tartanair_v2/House/Data_easy/P004",
        )
    ).expanduser().resolve()


def _artifact_root() -> Path:
    raw = os.environ.get("PIPELINE_TEASER_ARTIFACT_ROOT")
    if not raw:
        raise RuntimeError("PIPELINE_TEASER_ARTIFACT_ROOT is required")
    path = Path(raw).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_stereo_audit() -> list[torch.Tensor]:
    root = _data_root()
    result = audit.compute_audit(root)
    out = _artifact_root() / "stereo_pose_audit.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    b = result["baseline_m"]
    v = result["variation_vs_frame0"]
    t = result["translation_cv_m"]
    print(
        "[P004 pose-file stereo audit] "
        f"N={result['num_frames']} "
        f"baseline mean={b['mean']:.9f}m std={b['std']:.3e} "
        f"range=[{b['min']:.9f},{b['max']:.9f}]m "
        f"mean_t_cv=[{t['x']['mean']:.9f},{t['y']['mean']:.9f},{t['z']['mean']:.9f}] "
        f"max_delta_vs_f0={v['translation_delta_m']['max']:.3e}m/"
        f"{v['rotation_delta_deg']['max']:.3e}deg",
        flush=True,
    )
    print(f"[P004 pose-file stereo audit] saved: {out}", flush=True)

    left = audit.load_pose_rows(root / "pose_lcam_front.txt")
    right = audit.load_pose_rows(root / "pose_rcam_front.txt")
    return [torch.linalg.inv(left[i]) @ right[i] for i in range(min(len(left), len(right)))]


def _install_exact_posefile_stereo(relative_rows: list[torch.Tensor]) -> None:
    from async_pipeline.resplat_runtime import ResplatPacketGenerator

    original_make_batch = ResplatPacketGenerator._make_batch
    if getattr(original_make_batch, "_p004_exact_posefile_stereo", False):
        return

    def make_batch_exact_stereo(self, frame_input, T_left_c2w):
        batch = original_make_batch(self, frame_input, T_left_c2w)
        frame_index = int(frame_input.descriptor.frame_index)
        if frame_index < 0 or frame_index >= len(relative_rows):
            raise IndexError(
                f"frame {frame_index} outside exact stereo pose rows [0,{len(relative_rows)})"
            )

        relative = relative_rows[frame_index].to(dtype=torch.float32)
        T_left = T_left_c2w.detach().cpu().float()
        T_right = T_left @ relative

        batch["context"]["extrinsics"][0, 0] = T_left
        batch["context"]["extrinsics"][0, 1] = T_right
        batch["target"]["extrinsics"][0, 0] = T_left
        return batch

    make_batch_exact_stereo._p004_exact_posefile_stereo = True  # type: ignore[attr-defined]
    ResplatPacketGenerator._make_batch = make_batch_exact_stereo

    first = relative_rows[0]
    t = first[:3, 3]
    print(
        "[P004 pose-file stereo] enabled: exact per-frame full relative extrinsic; "
        f"frame0 t_cv={t.tolist()} baseline={float(torch.linalg.vector_norm(t)):.9f}m",
        flush=True,
    )


if __name__ == "__main__":
    rows = _write_stereo_audit()
    _install_exact_posefile_stereo(rows)
    p004._install_packet_artifacts()
    p004._install_final_artifacts()
    safe.repro.main()

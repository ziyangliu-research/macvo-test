#!/usr/bin/env python3
"""P004 full teaser run + 10k post-hoc native GraphDECO refinement.

The online stage is exactly the existing P004 qualitative teaser pipeline.  At
its endpoint we preserve the online artifacts, then run the already-validated
native GraphDECO post-hoc global-refinement schedule for 10k optimizer updates.
Finally we save a refined wide-view render at the same first-camera pose.  The
final refined PLY is saved by the native refinement wrapper and can be opened
with the existing headless web pose tuner (which automatically selects the
latest point_cloud/iteration_*/point_cloud.ply).
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torchvision.utils import save_image

import run_pipeline_execution_benchmark_repro_empty_safe as safe
import run_pipeline_execution_benchmark_repro_p004_teaser as p004
import run_pipeline_execution_benchmark_repro_posthoc_global_refine_native3dgs_lpips as native


def _install_refined_teaser_artifacts() -> None:
    from async_pipeline.backend_evaluation import BackendEvaluationMixin

    original_finalize = BackendEvaluationMixin._finalize_impl
    if getattr(original_finalize, "_p004_gr10k_refined_teaser_artifacts", False):
        return

    @torch.no_grad()
    def finalize_with_refined_teaser(self):
        # This call executes, in order:
        #   P004 online-artifact save -> native 10k refinement -> refined PLY save.
        summary = original_finalize(self)

        root = p004._artifact_root()
        refined_root = root / "teaser_overview_refined_10k"
        refined_root.mkdir(parents=True, exist_ok=True)

        if (
            self.gaussians is None
            or int(self.gaussians.get_xyz.shape[0]) == 0
            or not self.train_cameras
        ):
            meta = {
                "status": "EMPTY_MAP_OR_NO_CAMERA",
                "global_iteration": int(getattr(self, "global_iteration", 0)),
                "num_gaussians": 0
                if self.gaussians is None
                else int(self.gaussians.get_xyz.shape[0]),
            }
            (refined_root / "metadata.json").write_text(
                json.dumps(meta, indent=2), encoding="utf-8"
            )
            return summary

        first_twc = p004._first_camera_twc(self.train_cameras[0])
        camera = p004._make_wide_camera(self, first_twc)
        rendered = self.render(
            camera,
            self.gaussians,
            self.pipe,
            self.background,
            use_trained_exp=False,
            separate_sh=False,
        )["render"].clamp(0.0, 1.0)
        if torch.cuda.is_available():
            torch.cuda.current_stream(self.device).synchronize()

        save_image(rendered.detach().cpu(), refined_root / "render.png")
        (refined_root / "probe_pose_current_relative.json").write_text(
            json.dumps(first_twc.tolist(), indent=2), encoding="utf-8"
        )

        # Also place the same starting pose under teaser_overview so the existing
        # web tuner can be used with only --run_dir.  This pose is unchanged by
        # global refinement because camera poses are fixed.
        start_root = root / "teaser_overview"
        start_root.mkdir(parents=True, exist_ok=True)
        (start_root / "probe_pose_current_relative.json").write_text(
            json.dumps(first_twc.tolist(), indent=2), encoding="utf-8"
        )

        meta = {
            "status": "OK",
            "stage": "after_10000_native_graphdeco_global_refinement_updates",
            "pose_source": "first mapping camera; poses fixed during refinement",
            "frame_index": int(self.train_cameras[0].frame_index),
            "Twc": first_twc.tolist(),
            "render_width": 960,
            "render_height": 540,
            "fx_norm": 0.25,
            "fy_norm": 0.35,
            "cx_norm": 0.5,
            "cy_norm": 0.5,
            "global_iteration": int(self.global_iteration),
            "num_gaussians": int(self.gaussians.get_xyz.shape[0]),
        }
        (refined_root / "metadata.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
        print(
            "[P004 GR10k teaser] refined wide render saved: "
            f"{refined_root / 'render.png'} | "
            f"G={meta['num_gaussians']:,} global_iter={meta['global_iteration']}",
            flush=True,
        )
        return summary

    finalize_with_refined_teaser._p004_gr10k_refined_teaser_artifacts = True  # type: ignore[attr-defined]
    BackendEvaluationMixin._finalize_impl = finalize_with_refined_teaser


if __name__ == "__main__":
    # Keep raw packet artifacts from the existing P004 teaser run.
    p004._install_packet_artifacts()

    # Install native refinement first.  The P004 artifact wrapper installed next
    # will therefore save the ONLINE endpoint before delegating into refinement.
    native._install_native_graphdeco_refinement()
    p004._install_final_artifacts()

    # Outer wrapper: after the above chain returns, the map is the refined map.
    _install_refined_teaser_artifacts()
    safe.repro.main()

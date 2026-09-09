#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Sequence

import torch
from torchvision.utils import save_image


def install_online_test_render_export() -> None:
    """Save the online-endpoint render/GT pair for every held-out test view.

    This wraps the currently installed BackendEvaluationMixin._evaluate.  Install
    it *after* the LPIPS evaluator patch and before the native-refinement finalize
    patch.  Intermediate online evaluation is disabled in the formal launchers,
    so the first evaluation of ``self.test_cameras`` is the online endpoint,
    before the extra post-hoc opacity reset/refinement begins.
    """
    from async_pipeline.backend_evaluation import BackendEvaluationMixin

    original_evaluate = BackendEvaluationMixin._evaluate
    if getattr(original_evaluate, "_online_test_render_export_patch", False):
        return

    enabled = os.environ.get("PIPELINE_SAVE_ONLINE_TEST_RENDERS", "1").strip().lower()
    save_enabled = enabled not in {"0", "false", "no", "off"}

    @torch.no_grad()
    def evaluate_with_online_test_export(self, cameras: Sequence[Any]):
        metrics = original_evaluate(self, cameras)

        if not save_enabled:
            return metrics
        if getattr(self, "_online_test_render_export_done", False):
            return metrics
        if cameras is not self.test_cameras:
            return metrics

        root = Path(self.output_dir) / "online_test_renders"
        root.mkdir(parents=True, exist_ok=True)
        manifest: list[dict[str, Any]] = []

        print(
            f"[online-test-render] saving {len(self.test_cameras)} held-out views to {root}",
            flush=True,
        )

        for camera in self.test_cameras:
            frame_index = int(camera.frame_index)
            sequence_index = int(camera.sequence_index)
            view_dir = root / f"{frame_index:06d}"
            view_dir.mkdir(parents=True, exist_ok=True)

            render = self.render(
                camera,
                self.gaussians,
                self.pipe,
                self.background,
                use_trained_exp=False,
                separate_sh=False,
            )["render"].clamp(0.0, 1.0)
            gt = camera.original_image.clamp(0.0, 1.0)

            save_image(gt.detach().cpu(), view_dir / "gt.png")
            save_image(render.detach().cpu(), view_dir / "render.png")

            manifest.append(
                {
                    "frame_index": frame_index,
                    "sequence_index": sequence_index,
                    "folder": view_dir.name,
                    "gt": "gt.png",
                    "render": "render.png",
                }
            )

        (root / "manifest.json").write_text(
            json.dumps(
                {
                    "stage": "online_endpoint_before_global_refinement",
                    "index_convention": "dataset/frame_index, zero-based; same index used by the pipeline split",
                    "num_test_views": len(manifest),
                    "views": manifest,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        self._online_test_render_export_done = True
        print("[online-test-render] done", flush=True)
        return metrics

    evaluate_with_online_test_export._online_test_render_export_patch = True  # type: ignore[attr-defined]
    BackendEvaluationMixin._evaluate = evaluate_with_online_test_export

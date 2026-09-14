#!/usr/bin/env python3
"""Reproducible serial benchmark with final held-out Test PSNR only.

This wrapper keeps the online mapping algorithm unchanged and replaces the
normal finalize-time evaluation with a single metric: mean PSNR over all
held-out test cameras. It performs no post-hoc/global refinement and skips
train/active-map evaluation, SSIM, LPIPS, ATE, and image export.

The online optimizer uses the same empty-map-safe replay wrapper as the formal
native-3DGS benchmark. Aggressive opacity pruning is therefore allowed to prune
all Gaussians: meaningless zero-Gaussian backward passes are skipped until the
next train packet appends Gaussians again. Pruning itself is not weakened and no
Gaussian is rescued.
"""
from __future__ import annotations

import time

import torch

import run_pipeline_execution_benchmark_repro_empty_safe as safe


def _install_test_psnr_only_finalize() -> None:
    from async_pipeline.backend_evaluation import BackendEvaluationMixin

    if getattr(BackendEvaluationMixin._finalize_impl, "_test_psnr_only_patch", False):
        return

    @torch.no_grad()
    def finalize_test_psnr_only(self):
        self.initialize()

        from utils.image_utils import psnr

        values = []
        if self.gaussians is not None and int(self.gaussians.get_xyz.shape[0]) > 0:
            for camera in self.test_cameras:
                image = self.render(
                    camera,
                    self.gaussians,
                    self.pipe,
                    self.background,
                    use_trained_exp=False,
                    separate_sh=False,
                )["render"].clamp(0.0, 1.0)
                gt = camera.original_image.clamp(0.0, 1.0)
                values.append(float(psnr(image, gt).mean().item()))

        test_metrics = {
            "num_views": len(values),
            "psnr": (sum(values) / len(values)) if values else None,
        }
        payload = {
            "protocol": "online endpoint only; strict held-out Test PSNR",
            "num_train_packets": int(self.train_packet_count),
            "num_test_views": len(self.test_cameras),
            "global_iteration": int(self.global_iteration),
            "num_gaussians": (
                0 if self.gaussians is None else int(self.gaussians.get_xyz.shape[0])
            ),
            "test": test_metrics,
        }
        self._save_json("test_psnr_only.json", payload)

        psnr_value = test_metrics["psnr"]
        if psnr_value is None:
            print("[test-psnr-only] Test PSNR=MISSING (0 views)", flush=True)
        else:
            print(
                f"[test-psnr-only] views={len(values)} PSNR={psnr_value:.6f} dB "
                f"G={payload['num_gaussians']}",
                flush=True,
            )

        summary = {
            "backend": "StreamingIncrementalBackend",
            "num_train_packets": int(self.train_packet_count),
            "num_train_cameras": len(self.train_cameras),
            "num_test_cameras": len(self.test_cameras),
            "total_iterations": int(self.global_iteration),
            "final_num_gaussians": payload["num_gaussians"],
            "final_metrics": {"test_all": test_metrics},
            "gpu_memory": self._gpu_memory_stats(),
            "wall_time_sec": time.perf_counter() - self.wall_start,
        }
        self._save_json("incremental_backend_summary.json", summary)
        if self.wandb_run is not None:
            self.wandb_run.finish()
        return summary

    finalize_test_psnr_only._test_psnr_only_patch = True  # type: ignore[attr-defined]
    BackendEvaluationMixin._finalize_impl = finalize_test_psnr_only


if __name__ == "__main__":
    _install_test_psnr_only_finalize()
    safe.repro.main()

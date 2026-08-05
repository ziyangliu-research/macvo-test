from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional, Sequence

import torch

from .backend_camera import StreamingCamera


class BackendEvaluationMixin:
    @torch.no_grad()
    def _evaluate(self, cameras: Sequence[StreamingCamera]) -> dict[str, float | int]:
        from utils.image_utils import psnr
        from utils.loss_utils import l1_loss, ssim

        selected = list(cameras)
        if self.config.eval_max_views > 0:
            selected = selected[: self.config.eval_max_views]
        if not selected:
            return {"num_views": 0}
        l1_sum = psnr_sum = ssim_sum = 0.0
        for camera in selected:
            image = self.render(
                camera,
                self.gaussians,
                self.pipe,
                self.background,
                use_trained_exp=False,
                separate_sh=False,
            )["render"].clamp(0.0, 1.0)
            gt = camera.original_image.clamp(0.0, 1.0)
            l1_sum += float(l1_loss(image, gt).mean().item())
            psnr_sum += float(psnr(image, gt).mean().item())
            ssim_sum += float(ssim(image, gt).mean().item())
        count = len(selected)
        return {
            "num_views": count,
            "l1": l1_sum / count,
            "psnr": psnr_sum / count,
            "ssim": ssim_sum / count,
        }

    def _record_evaluation(
        self,
        stage: str,
        update: BackendUpdate,
        active_cameras: Sequence[StreamingCamera],
    ) -> None:
        seen_test = [
            camera
            for camera in self.test_cameras
            if camera.sequence_index <= update.descriptor.sequence_index
        ]
        start = time.perf_counter()
        entry = {
            "stage": stage,
            "frame_index": update.descriptor.frame_index,
            "sequence_index": update.descriptor.sequence_index,
            "global_iteration": self.global_iteration,
            "train_packet_count": self.train_packet_count,
            "num_gaussians": int(self.gaussians.get_xyz.shape[0]),
            "train_inserted": self._evaluate(self.train_cameras),
            "active_local_map": self._evaluate(active_cameras),
            "test_seen": self._evaluate(seen_test),
        }
        entry["eval_time_sec"] = self._sync_elapsed(start)
        self.metrics_log.append(entry)
        if self.config.write_runtime_artifacts:
            self._save_json("metrics_log.json", self.metrics_log)
        if self.wandb_run is not None:
            values: dict[str, float] = {
                "eval/num_gaussians": float(entry["num_gaussians"]),
                "eval/frame_index": float(update.descriptor.frame_index),
            }
            for split in ("train_inserted", "active_local_map", "test_seen"):
                metrics = entry[split]
                for name in ("l1", "psnr", "ssim"):
                    if name in metrics:
                        values[f"eval/{split}_{name}"] = float(metrics[name])
            self.wandb_run.log(values, step=self.global_iteration)

    def _log_wandb_packet(self, timing: dict[str, Any]) -> None:
        if self.wandb_run is None:
            return
        self.wandb_run.log(
            {
                "timing/backend_total_sec": timing["backend_total_sec"],
                "timing/resplat_inference_sec": timing["resplat_inference_sec"],
                "timing/pose_latency_sec": timing["pose_latency_sec"],
                "timing/join_wait_sec": timing["join_wait_sec"],
                "scene/num_gaussians": timing["num_gaussians"],
                "stream/train_packet_count": timing["train_packet_count"],
            },
            step=self.global_iteration,
        )

    def _save_point_cloud(self, iteration: int) -> Optional[Path]:
        if self.gaussians is None or int(self.gaussians.get_xyz.shape[0]) == 0:
            return None
        path = (
            self.output_dir
            / "point_cloud"
            / f"iteration_{iteration}"
            / "point_cloud.ply"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        self.gaussians.save_ply(str(path))
        return path

    def _save_json(self, name: str, value: Any) -> None:
        (self.output_dir / name).write_text(
            json.dumps(value, indent=2), encoding="utf-8"
        )

    def _sync_elapsed(self, start: float) -> float:
        if self.stream is not None:
            self.stream.synchronize()
        else:
            torch.cuda.current_stream(self.device).synchronize()
        return time.perf_counter() - start

    def finalize(self) -> dict[str, Any]:
        self.initialize()
        if self.stream is None:
            return self._finalize_impl()
        with torch.cuda.stream(self.stream):
            summary = self._finalize_impl()
        self.stream.synchronize()
        return summary

    def _finalize_impl(self) -> dict[str, Any]:
        final_metrics: dict[str, Any] = {}
        if self.train_packet_count > 0:
            if self.config.evaluation_enabled:
                final_metrics = {
                    "train_inserted": self._evaluate(self.train_cameras),
                    "active_local_map": self._evaluate(
                        self.train_cameras[-self.config.local_map_size :]
                    ),
                    "test_all": self._evaluate(self.test_cameras),
                }
            if self.config.save_final_ply:
                self._save_point_cloud(self.global_iteration)
        summary = {
            "backend": "StreamingIncrementalBackend",
            "coordinate_contract": (
                "ReSplat packet is left-camera local; backend applies metric OpenCV Twc "
                "to means and rotations before insertion"
            ),
            "num_train_packets": self.train_packet_count,
            "num_train_cameras": len(self.train_cameras),
            "num_test_cameras": len(self.test_cameras),
            "total_iterations": self.global_iteration,
            "final_num_gaussians": (
                0
                if self.gaussians is None
                else int(self.gaussians.get_xyz.shape[0])
            ),
            "spatial_lr_scale": self.config.spatial_lr_scale,
            "num_skipped_invalid_poses": len(self.skipped_pose_log),
            "evaluation_enabled": self.config.evaluation_enabled,
            "write_runtime_artifacts": self.config.write_runtime_artifacts,
            "final_metrics": final_metrics,
            "wall_time_sec": time.perf_counter() - self.wall_start,
        }
        self._save_json("incremental_backend_summary.json", summary)
        if self.wandb_run is not None:
            self.wandb_run.finish()
        return summary

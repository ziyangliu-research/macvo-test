from __future__ import annotations

import time

import torch

from .backend_core import StreamingIncrementalBackend as _BaseStreamingBackend
from .contracts import BackendUpdate, LocalGaussianPacket
from .geometry import rotate_harmonics_local_to_world


class StreamingIncrementalBackend(_BaseStreamingBackend):
    """GraphDECO backend with SH alignment and visible packet-level progress."""

    def __init__(self, config) -> None:
        super().__init__(config)
        self._progress_start = time.perf_counter()

    def _packet_to_graphdeco(
        self,
        packet: LocalGaussianPacket,
        T_world_from_left: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], dict[str, float | bool]]:
        tensors, reset_stats = super()._packet_to_graphdeco(
            packet, T_world_from_left
        )

        non_blocking = (
            packet.harmonics.device.type == "cpu"
            and packet.harmonics.is_pinned()
        )
        harmonics_local = packet.harmonics.to(
            self.device, dtype=torch.float32, non_blocking=non_blocking
        )
        T = T_world_from_left.to(self.device, dtype=torch.float32)
        harmonics_world = rotate_harmonics_local_to_world(harmonics_local, T)

        target_coeffs = (self.config.sh_degree + 1) ** 2
        source_coeffs = int(harmonics_world.shape[-1])
        if source_coeffs < target_coeffs:
            pad = torch.zeros(
                harmonics_world.shape[0],
                3,
                target_coeffs - source_coeffs,
                device=self.device,
                dtype=harmonics_world.dtype,
            )
            harmonics_world = torch.cat([harmonics_world, pad], dim=-1)
        elif source_coeffs > target_coeffs:
            harmonics_world = harmonics_world[..., :target_coeffs]

        tensors["f_dc"] = harmonics_world[:, :, 0].unsqueeze(1).contiguous()
        tensors["f_rest"] = (
            harmonics_world[:, :, 1:].permute(0, 2, 1).contiguous()
        )
        return tensors, reset_stats

    def _process_impl(self, update: BackendUpdate) -> None:
        """Run the normal update and print one compact, flushed progress line."""

        before_packets = self.train_packet_count
        start = time.perf_counter()
        super()._process_impl(update)
        update_sec = time.perf_counter() - start

        if update.descriptor.is_test:
            print(
                f"[backend] test frame={update.descriptor.frame_index:04d} "
                f"registered ({len(self.test_cameras)} test views)",
                flush=True,
            )
            return

        if self.train_packet_count == before_packets:
            # Invalid-pose skip or another non-insertion path.
            print(
                f"[backend] train frame={update.descriptor.frame_index:04d} skipped",
                flush=True,
            )
            return

        completed = self.train_packet_count
        configured_total_iterations = int(self.config.optimization.iterations)
        expected_packets = max(
            completed,
            configured_total_iterations // self.config.iterations_per_packet,
        )
        elapsed = time.perf_counter() - self._progress_start
        average = elapsed / max(completed, 1)
        remaining = max(expected_packets - completed, 0)
        eta = average * remaining
        num_gaussians = int(self.gaussians.get_xyz.shape[0])

        metric_text = ""
        if self.metrics_log:
            latest = self.metrics_log[-1]
            if latest.get("frame_index") == update.descriptor.frame_index:
                test_metric = latest.get("test_seen", {})
                train_metric = latest.get("train_inserted", {})
                pieces: list[str] = []
                if "psnr" in train_metric:
                    pieces.append(f"trainPSNR={float(train_metric['psnr']):.2f}")
                if "psnr" in test_metric:
                    pieces.append(f"testPSNR={float(test_metric['psnr']):.2f}")
                if pieces:
                    metric_text = " | " + " ".join(pieces)

        resource_text = ""
        if self.timing_log:
            latest_timing = self.timing_log[-1]
            if latest_timing.get("frame_index") == update.descriptor.frame_index:
                resource_text = (
                    " | VRAM="
                    f"{float(latest_timing['gpu_memory_allocated_gb']):.2f}GB "
                    f"reserved={float(latest_timing['gpu_memory_reserved_gb']):.2f}GB "
                    f"peak={float(latest_timing['gpu_peak_memory_allocated_gb']):.2f}GB "
                    f"free={float(latest_timing['gpu_free_memory_gb']):.2f}GB"
                )

        print(
            f"[backend] packet {completed:02d}/{expected_packets:02d} "
            f"frame={update.descriptor.frame_index:04d} "
            f"iter={self.global_iteration}/{configured_total_iterations} "
            f"G={num_gaussians:,} "
            f"last={update_sec:.2f}s elapsed={elapsed:.1f}s ETA={eta:.1f}s"
            f"{metric_text}"
            f"{resource_text}",
            flush=True,
        )

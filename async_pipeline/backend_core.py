from __future__ import annotations

import random
import sys
import time
from types import SimpleNamespace
from typing import Any, Optional, Sequence

import torch
from torch import nn

from .backend_camera import StreamingCamera
from .backend_config import StreamingBackendConfig
from .backend_evaluation import BackendEvaluationMixin
from .contracts import BackendUpdate, LocalGaussianPacket
from .geometry import (
    quaternion_xyzw_to_wxyz,
    rotate_local_quaternions_to_world_xyzw,
)


class StreamingIncrementalBackend(BackendEvaluationMixin):
    def __init__(self, config: StreamingBackendConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.output_dir = config.output_dir.expanduser().resolve()
        self.gaussians: Any = None
        self.render: Any = None
        self.get_projection_matrix: Any = None
        self.pipe: Any = None
        self.opt = config.optimization.namespace()
        self.background: Optional[torch.Tensor] = None
        self.train_cameras: list[StreamingCamera] = []
        self.test_cameras: list[StreamingCamera] = []
        self.global_iteration = 0
        self.train_packet_count = 0
        self.metrics_log: list[dict[str, Any]] = []
        self.timing_log: list[dict[str, Any]] = []
        self.maintenance_log: list[dict[str, Any]] = []
        self.skipped_pose_log: list[dict[str, Any]] = []
        self.wandb_run: Any = None
        self.wall_start = 0.0
        self._initialized = False
        self.stream: Optional[torch.cuda.Stream] = None

    def initialize(self) -> None:
        if self._initialized:
            return
        repo = self.config.gs_repo.expanduser().resolve()
        if not repo.is_dir():
            raise FileNotFoundError(f"gaussian-splatting repository not found: {repo}")
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError(
                f"GraphDECO backend requires CUDA; device={self.device}, "
                f"cuda_available={torch.cuda.is_available()}"
            )
        torch.cuda.set_device(self.device)
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        from gaussian_renderer import render
        from scene import GaussianModel
        from utils.graphics_utils import getProjectionMatrix

        self.render = render
        self.get_projection_matrix = getProjectionMatrix
        self.gaussians = GaussianModel(
            self.config.sh_degree, self.opt.optimizer_type
        )
        self.pipe = SimpleNamespace(
            convert_SHs_python=False,
            compute_cov3D_python=False,
            debug=False,
            antialiasing=self.config.antialiasing,
        )
        color = [1.0, 1.0, 1.0] if self.config.white_background else [0.0, 0.0, 0.0]
        self.background = torch.tensor(color, dtype=torch.float32, device=self.device)
        if self.config.dedicated_cuda_stream:
            current = torch.cuda.current_stream(self.device)
            self.stream = torch.cuda.Stream(device=self.device)
            self.stream.wait_stream(current)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.wall_start = time.perf_counter()
        self._init_wandb()
        torch.cuda.reset_peak_memory_stats(self.device)
        self._initialized = True

    def _init_wandb(self) -> None:
        if self.config.wandb_mode == "disabled":
            return
        try:
            import wandb
        except ImportError:
            return
        self.wandb_run = wandb.init(
            project=self.config.wandb_project,
            name=self.config.wandb_run_name or self.output_dir.name,
            mode=self.config.wandb_mode,
            config={
                "backend": self.config.__dict__,
                "optimization": self.config.optimization.__dict__,
            },
        )

    def _gpu_memory_stats(self) -> dict[str, float]:
        """Return packet-aligned PyTorch and device-level GPU memory statistics."""

        gb = float(1024**3)
        allocated = torch.cuda.memory_allocated(self.device)
        reserved = torch.cuda.memory_reserved(self.device)
        peak_allocated = torch.cuda.max_memory_allocated(self.device)
        peak_reserved = torch.cuda.max_memory_reserved(self.device)
        free_bytes, total_bytes = torch.cuda.mem_get_info(self.device)
        return {
            "gpu_memory_allocated_gb": float(allocated) / gb,
            "gpu_memory_reserved_gb": float(reserved) / gb,
            "gpu_peak_memory_allocated_gb": float(peak_allocated) / gb,
            "gpu_peak_memory_reserved_gb": float(peak_reserved) / gb,
            "gpu_free_memory_gb": float(free_bytes) / gb,
            "gpu_total_memory_gb": float(total_bytes) / gb,
            "gpu_used_global_gb": float(total_bytes - free_bytes) / gb,
        }

    def process(self, update: BackendUpdate) -> None:
        self.initialize()
        if self.stream is None:
            self._process_impl(update)
            return
        with torch.cuda.stream(self.stream):
            self._process_impl(update)

    def _process_impl(self, update: BackendUpdate) -> None:
        update.validate()
        if not update.pose.valid:
            event = {
                "sequence_index": update.descriptor.sequence_index,
                "frame_index": update.descriptor.frame_index,
                "reason": "MAC-VO need_interp/uncommitted valid pose unavailable online",
                "policy": self.config.invalid_pose_policy,
            }
            if self.config.invalid_pose_policy == "error":
                raise RuntimeError(f"invalid MAC-VO pose: {event}")
            self.skipped_pose_log.append(event)
            if self.config.write_runtime_artifacts:
                self._save_json("skipped_invalid_poses.json", self.skipped_pose_log)
            return
        camera = StreamingCamera(update, self.get_projection_matrix, self.device)
        if update.descriptor.is_test:
            self.test_cameras.append(camera)
            return
        assert update.packet is not None
        packet_start = time.perf_counter()
        tensors, reset_stats = self._packet_to_graphdeco(
            update.packet, update.pose.T_world_from_left
        )
        conversion_sec = self._sync_elapsed(packet_start)

        append_start = time.perf_counter()
        if self.train_packet_count == 0:
            self._initialize_gaussians(tensors)
        else:
            self._append_gaussians(tensors)
        append_sec = self._sync_elapsed(append_start)

        self.train_cameras.append(camera)
        self.train_packet_count += 1
        active_cameras = self.train_cameras[-self.config.local_map_size :]
        should_eval = self.config.evaluation_enabled and (
            self.train_packet_count == 1
            or self.train_packet_count % self.config.eval_every_train_packets == 0
        )

        if self.config.eval_before_optimization and should_eval:
            self._record_evaluation(
                stage=f"packet_{update.descriptor.frame_index:06d}_before",
                update=update,
                active_cameras=active_cameras,
            )

        optimization_start = time.perf_counter()
        maintenance_event = self._optimize_active_map(
            update=update,
            active_cameras=active_cameras,
        )
        optimization_sec = self._sync_elapsed(optimization_start)

        timing = {
            "sequence_index": update.descriptor.sequence_index,
            "frame_index": update.descriptor.frame_index,
            "train_packet_count": self.train_packet_count,
            "num_gaussians": int(self.gaussians.get_xyz.shape[0]),
            "packet_conversion_alignment_sec": conversion_sec,
            "packet_append_sec": append_sec,
            "local_optimization_and_maintenance_sec": optimization_sec,
            "backend_total_sec": conversion_sec + append_sec + optimization_sec,
            "resplat_inference_sec": update.packet.inference_sec,
            "pose_latency_sec": update.pose.latency_sec,
            "join_wait_sec": update.join_wait_sec,
            "new_packet_opacity_reset": reset_stats,
            "maintenance": maintenance_event,
        }
        timing.update(self._gpu_memory_stats())
        self.timing_log.append(timing)
        if self.config.write_runtime_artifacts:
            self._save_json("timing_log.json", self.timing_log)

        if should_eval:
            self._record_evaluation(
                stage=f"packet_{update.descriptor.frame_index:06d}_after",
                update=update,
                active_cameras=active_cameras,
            )
        if (
            self.config.save_every_train_packets > 0
            and self.train_packet_count % self.config.save_every_train_packets == 0
        ):
            self._save_point_cloud(self.global_iteration)
        self._log_wandb_packet(timing)

    def _packet_to_graphdeco(
        self,
        packet: LocalGaussianPacket,
        T_world_from_left: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], dict[str, float | bool]]:
        packet.validate()
        if packet.coordinate_frame != "left_camera_local":
            raise ValueError(
                f"backend expects left-local packets, got {packet.coordinate_frame}"
            )
        non_blocking = packet.means.device.type == "cpu" and packet.means.is_pinned()
        means_local = packet.means.to(
            self.device, dtype=torch.float32, non_blocking=non_blocking
        )
        scales = packet.scales.to(
            self.device, dtype=torch.float32, non_blocking=non_blocking
        )
        rotations_local_xyzw = packet.rotations_xyzw.to(
            self.device, dtype=torch.float32, non_blocking=non_blocking
        )
        harmonics = packet.harmonics.to(
            self.device, dtype=torch.float32, non_blocking=non_blocking
        )
        opacity = packet.opacities.reshape(-1, 1).to(
            self.device, dtype=torch.float32, non_blocking=non_blocking
        )
        T = T_world_from_left.to(self.device, dtype=torch.float32)
        R, t = T[:3, :3], T[:3, 3]
        means_world = means_local @ R.transpose(0, 1) + t
        rotations_world_xyzw = rotate_local_quaternions_to_world_xyzw(
            rotations_local_xyzw, T
        )
        rotations_world_wxyz = quaternion_xyzw_to_wxyz(rotations_world_xyzw)

        opacity_before = opacity
        if self.config.reset_new_packet_opacity:
            opacity = torch.minimum(
                opacity,
                torch.full_like(
                    opacity, self.config.new_packet_reset_max_opacity
                ),
            )
        eps = 1e-6
        opacity_internal = torch.logit(opacity.clamp(eps, 1.0 - eps))
        scale_internal = torch.log(scales.clamp_min(1e-8))

        target_coeffs = (self.config.sh_degree + 1) ** 2
        source_coeffs = int(harmonics.shape[-1])
        if source_coeffs < target_coeffs:
            pad = torch.zeros(
                harmonics.shape[0],
                3,
                target_coeffs - source_coeffs,
                device=self.device,
                dtype=harmonics.dtype,
            )
            harmonics = torch.cat([harmonics, pad], dim=-1)
        elif source_coeffs > target_coeffs:
            harmonics = harmonics[..., :target_coeffs]
        f_dc = harmonics[:, :, 0].unsqueeze(1).contiguous()
        f_rest = harmonics[:, :, 1:].permute(0, 2, 1).contiguous()
        tensors = {
            "xyz": means_world.contiguous(),
            "f_dc": f_dc,
            "f_rest": f_rest,
            "opacity": opacity_internal.contiguous(),
            "scaling": scale_internal.contiguous(),
            "rotation": rotations_world_wxyz.contiguous(),
        }
        reset_stats: dict[str, float | bool] = {
            "enabled": self.config.reset_new_packet_opacity,
            "before_mean": float(opacity_before.mean().item()),
            "before_max": float(opacity_before.max().item()),
            "after_mean": float(opacity.mean().item()),
            "after_max": float(opacity.max().item()),
        }
        return tensors, reset_stats

    def _initialize_gaussians(self, tensors: dict[str, torch.Tensor]) -> None:
        g = self.gaussians
        g.spatial_lr_scale = float(self.config.spatial_lr_scale)
        g._xyz = nn.Parameter(tensors["xyz"].requires_grad_(True))
        g._features_dc = nn.Parameter(tensors["f_dc"].requires_grad_(True))
        g._features_rest = nn.Parameter(tensors["f_rest"].requires_grad_(True))
        g._opacity = nn.Parameter(tensors["opacity"].requires_grad_(True))
        g._scaling = nn.Parameter(tensors["scaling"].requires_grad_(True))
        g._rotation = nn.Parameter(tensors["rotation"].requires_grad_(True))
        g.active_sh_degree = g.max_sh_degree
        count = int(g._xyz.shape[0])
        g.max_radii2D = torch.zeros(count, device=self.device)
        g.xyz_gradient_accum = torch.zeros((count, 1), device=self.device)
        g.denom = torch.zeros((count, 1), device=self.device)
        g.exposure_mapping = {}
        g.pretrained_exposures = None
        g._exposure = nn.Parameter(
            torch.eye(3, 4, device=self.device).unsqueeze(0).requires_grad_(True)
        )
        g.training_setup(self.opt)

    def _append_gaussians(self, tensors: dict[str, torch.Tensor]) -> None:
        g = self.gaussians
        appended = g.cat_tensors_to_optimizer(tensors)
        g._xyz = appended["xyz"]
        g._features_dc = appended["f_dc"]
        g._features_rest = appended["f_rest"]
        g._opacity = appended["opacity"]
        g._scaling = appended["scaling"]
        g._rotation = appended["rotation"]
        count = int(tensors["xyz"].shape[0])
        g.max_radii2D = torch.cat(
            [g.max_radii2D, torch.zeros(count, device=self.device)]
        )
        g.xyz_gradient_accum = torch.cat(
            [g.xyz_gradient_accum, torch.zeros((count, 1), device=self.device)]
        )
        g.denom = torch.cat(
            [g.denom, torch.zeros((count, 1), device=self.device)]
        )

    def _optimize_active_map(
        self,
        update: BackendUpdate,
        active_cameras: Sequence[StreamingCamera],
    ) -> Optional[dict[str, Any]]:
        from utils.loss_utils import l1_loss, ssim

        maintenance_event: Optional[dict[str, Any]] = None
        stack = list(active_cameras)
        for local_iteration in range(1, self.config.iterations_per_packet + 1):
            self.global_iteration += 1
            self.gaussians.update_learning_rate(self.global_iteration)
            if not stack:
                stack = list(active_cameras)
            camera = stack.pop(random.randrange(len(stack)))
            background = (
                torch.rand(3, device=self.device)
                if self.opt.random_background
                else self.background
            )
            render_pkg = self.render(
                camera,
                self.gaussians,
                self.pipe,
                background,
                use_trained_exp=False,
                separate_sh=False,
            )
            image = render_pkg["render"]
            gt = camera.original_image
            if camera.alpha_mask is not None:
                image = image * camera.alpha_mask
            ll1 = l1_loss(image, gt)
            ssim_value = ssim(image, gt)
            loss = (
                (1.0 - self.opt.lambda_dssim) * ll1
                + self.opt.lambda_dssim * (1.0 - ssim_value)
            )
            loss.backward()

            with torch.no_grad():
                collecting = (
                    self.config.maintenance_mode == "standard"
                    and local_iteration
                    <= self.config.maintenance_after_local_iteration
                )
                if collecting:
                    indices = self._visibility_indices(
                        render_pkg["visibility_filter"],
                        int(self.gaussians.get_xyz.shape[0]),
                    )
                    radii = render_pkg["radii"]
                    if indices.numel() > 0:
                        self.gaussians.max_radii2D[indices] = torch.maximum(
                            self.gaussians.max_radii2D[indices], radii[indices]
                        )
                        self.gaussians.add_densification_stats(
                            render_pkg["viewspace_points"], indices
                        )

                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)

                if (
                    self.config.maintenance_mode == "standard"
                    and local_iteration
                    == self.config.maintenance_after_local_iteration
                ):
                    maintenance_event = self._run_maintenance(
                        update,
                        local_iteration,
                        render_pkg["radii"],
                    )

                if (
                    self.wandb_run is not None
                    and self.global_iteration % self.config.wandb_log_interval == 0
                ):
                    self.wandb_run.log(
                        {
                            "train/loss": float(loss.item()),
                            "train/l1": float(ll1.item()),
                            "train/ssim": float(ssim_value.item()),
                            "scene/num_gaussians": int(
                                self.gaussians.get_xyz.shape[0]
                            ),
                            "stream/frame_index": update.descriptor.frame_index,
                            "stream/train_packet_count": self.train_packet_count,
                        },
                        step=self.global_iteration,
                    )
        return maintenance_event

    def _visibility_indices(
        self, value: torch.Tensor, count: int
    ) -> torch.Tensor:
        flat = value.reshape(-1)
        if flat.dtype == torch.bool:
            if flat.numel() != count:
                raise RuntimeError(
                    f"visibility mask size {flat.numel()} != Gaussian count {count}"
                )
            return flat.nonzero().reshape(-1)
        if flat.dtype.is_floating_point:
            if flat.numel() != count:
                raise RuntimeError("floating visibility tensor has unexpected size")
            return (flat > 0).nonzero().reshape(-1)
        indices = flat.long()
        if indices.numel() > 0 and (
            int(indices.min()) < 0 or int(indices.max()) >= count
        ):
            raise RuntimeError("visibility indices outside Gaussian range")
        return indices

    def _current_scene_extent(self) -> float:
        cameras = self.train_cameras
        if len(cameras) <= 1:
            return float(self.config.minimum_scene_extent)
        centers = torch.stack([camera.camera_center for camera in cameras], dim=0)
        center = centers.mean(dim=0)
        radius = torch.linalg.vector_norm(centers - center, dim=-1).max()
        return max(
            float(self.config.minimum_scene_extent),
            1.1 * float(radius.item()),
        )

    def _run_maintenance(
        self,
        update: BackendUpdate,
        local_iteration: int,
        radii: torch.Tensor,
    ) -> dict[str, Any]:
        count_before = int(self.gaussians.get_xyz.shape[0])
        threshold = (
            self.config.maintenance_grad_threshold
            if self.config.maintenance_grad_threshold > 0
            else self.opt.densify_grad_threshold
        )
        max_screen = (
            self.config.maintenance_max_screen_size
            if self.config.maintenance_max_screen_size > 0
            else None
        )
        extent = self._current_scene_extent()
        start = time.perf_counter()
        self.gaussians.densify_and_prune(
            threshold,
            self.config.maintenance_min_opacity,
            extent,
            max_screen,
            radii,
        )
        sec = self._sync_elapsed(start)
        event = {
            "frame_index": update.descriptor.frame_index,
            "global_iteration": self.global_iteration,
            "local_iteration": local_iteration,
            "count_before": count_before,
            "count_after": int(self.gaussians.get_xyz.shape[0]),
            "grad_threshold": threshold,
            "min_opacity": self.config.maintenance_min_opacity,
            "max_screen_size": max_screen,
            "scene_extent": extent,
            "maintenance_sec": sec,
        }
        self.maintenance_log.append(event)
        if self.config.write_runtime_artifacts:
            self._save_json("maintenance_log.json", self.maintenance_log)
        return event

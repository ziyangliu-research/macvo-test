from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace


@dataclass(frozen=True)
class BackendOptimizationConfig:
    iterations: int = 4000
    position_lr_init: float = 0.00016
    position_lr_final: float = 0.0000016
    position_lr_delay_mult: float = 0.01
    position_lr_max_steps: int = 30000
    feature_lr: float = 0.0025
    opacity_lr: float = 0.025
    scaling_lr: float = 0.005
    rotation_lr: float = 0.001
    exposure_lr_init: float = 0.01
    exposure_lr_final: float = 0.001
    exposure_lr_delay_steps: int = 0
    exposure_lr_delay_mult: float = 0.0
    percent_dense: float = 0.01
    lambda_dssim: float = 0.2
    densify_grad_threshold: float = 0.0002
    random_background: bool = False
    optimizer_type: str = "default"

    def namespace(self) -> SimpleNamespace:
        return SimpleNamespace(**self.__dict__)


@dataclass(frozen=True)
class StreamingBackendConfig:
    gs_repo: Path
    output_dir: Path
    device: str = "cuda:0"
    sh_degree: int = 3
    local_map_size: int = 10
    iterations_per_packet: int = 100
    reset_new_packet_opacity: bool = True
    new_packet_reset_max_opacity: float = 0.01
    maintenance_mode: str = "standard"
    maintenance_after_local_iteration: int = 50
    maintenance_grad_threshold: float = -1.0
    maintenance_min_opacity: float = 0.005
    maintenance_max_screen_size: float = -1.0
    spatial_lr_scale: float = 1.0
    minimum_scene_extent: float = 1.0
    white_background: bool = False
    antialiasing: bool = False
    eval_before_optimization: bool = True
    eval_every_train_packets: int = 1
    eval_max_views: int = 0
    save_every_train_packets: int = 5
    save_final_ply: bool = True
    wandb_mode: str = "disabled"
    wandb_project: str = "macvo-resplat-3dgs-async"
    wandb_run_name: str = ""
    wandb_log_interval: int = 10
    dedicated_cuda_stream: bool = True
    invalid_pose_policy: str = "skip"
    write_runtime_artifacts: bool = True
    evaluation_enabled: bool = True
    optimization: BackendOptimizationConfig = field(
        default_factory=BackendOptimizationConfig
    )

    def __post_init__(self) -> None:
        if self.sh_degree < 0:
            raise ValueError("sh_degree must be non-negative")
        if self.local_map_size <= 0:
            raise ValueError("local_map_size must be positive")
        if self.iterations_per_packet <= 0:
            raise ValueError("iterations_per_packet must be positive")
        if not 0 < self.new_packet_reset_max_opacity < 1:
            raise ValueError("new_packet_reset_max_opacity must be in (0,1)")
        if self.maintenance_mode not in {"off", "standard"}:
            raise ValueError("maintenance_mode must be off or standard")
        if self.maintenance_mode == "standard" and not (
            1
            <= self.maintenance_after_local_iteration
            <= self.iterations_per_packet
        ):
            raise ValueError(
                "standard maintenance iteration must lie in [1,iterations_per_packet]"
            )
        if self.spatial_lr_scale <= 0 or self.minimum_scene_extent <= 0:
            raise ValueError("spatial scales must be positive")
        if self.eval_every_train_packets <= 0:
            raise ValueError("eval_every_train_packets must be positive")
        if self.invalid_pose_policy not in {"skip", "error"}:
            raise ValueError("invalid_pose_policy must be skip or error")

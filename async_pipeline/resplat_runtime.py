from __future__ import annotations

import contextlib
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional, Sequence

import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms.functional as TF

from .contracts import FrameDescriptor, LocalGaussianPacket, StereoFrameInput
from .geometry import (
    fixed_tartanair_stereo_rig_cv,
    rotate_local_quaternions_to_world_xyzw,
)
from .resplat_packet import packet_from_resplat_gaussians


@contextlib.contextmanager
def temporary_cwd(path: Path):
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def _resolve_repo_relative(value: Any, repo: Path) -> Optional[Path]:
    if value is None:
        return None
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else (repo / path).resolve()


def _tensor_to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device=device, non_blocking=True)
    if isinstance(value, dict):
        return {key: _tensor_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_tensor_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_tensor_to_device(item, device) for item in value)
    return value


def build_pixel_intrinsic(fx: float, fy: float, cx: float, cy: float) -> torch.Tensor:
    if not all(value > 0 for value in (fx, fy, cx, cy)):
        raise ValueError(f"invalid intrinsics: fx={fx}, fy={fy}, cx={cx}, cy={cy}")
    K = torch.eye(3, dtype=torch.float32)
    K[0, 0], K[1, 1] = float(fx), float(fy)
    K[0, 2], K[1, 2] = float(cx), float(cy)
    return K


def process_pil_image_and_intrinsic(
    image: Image.Image,
    K_pixel: torch.Tensor,
    image_shape: tuple[int, int],
    normalize_intrinsics: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mirror the validated TartanAir ReSplat resize + center-crop path."""

    image = image.convert("RGB")
    original_w, original_h = image.size
    target_h, target_w = image_shape
    scale = max(target_w / original_w, target_h / original_h)
    resized_w = int(round(original_w * scale))
    resized_h = int(round(original_h * scale))
    image = image.resize((resized_w, resized_h), Image.Resampling.BILINEAR)
    left = int(round((resized_w - target_w) / 2.0))
    top = int(round((resized_h - target_h) / 2.0))
    image = image.crop((left, top, left + target_w, top + target_h))
    tensor = TF.to_tensor(image)

    K = K_pixel.clone().float()
    K[0, 0] *= scale
    K[1, 1] *= scale
    K[0, 2] = K[0, 2] * scale - left
    K[1, 2] = K[1, 2] * scale - top
    if normalize_intrinsics:
        K[0, 0] /= target_w
        K[1, 1] /= target_h
        K[0, 2] /= target_w
        K[1, 2] /= target_h
    return tensor, K


def process_image_and_intrinsic(
    path: Path,
    K_pixel: torch.Tensor,
    image_shape: tuple[int, int],
    normalize_intrinsics: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    with Image.open(path) as image_file:
        return process_pil_image_and_intrinsic(
            image_file, K_pixel, image_shape, normalize_intrinsics
        )


@dataclass(frozen=True)
class ResplatRuntimeConfig:
    repo: Path
    experiment: str
    device: str = "cuda:0"
    checkpoint: Optional[Path] = None
    overrides: tuple[str, ...] = ()
    output_dir: Path = Path("outputs/resplat_async_runtime")
    fx: float = 320.0
    fy: float = 320.0
    cx: float = 320.0
    cy: float = 320.0
    stereo_baseline: float = 0.25000006
    refine_steps: int = 0
    refine_use_target: bool = False
    deterministic: bool = False
    pin_output_memory: bool = True
    input_mode: Literal["shared_tensors", "file_paths"] = "shared_tensors"
    handoff_mode: Literal["pinned_cpu", "gpu"] = "pinned_cpu"
    strict_validation: bool = False

    def __post_init__(self) -> None:
        if not self.experiment:
            raise ValueError("experiment must not be empty")
        if self.refine_steps < 0:
            raise ValueError("refine_steps must be non-negative")
        if self.stereo_baseline <= 0:
            raise ValueError("stereo_baseline must be positive")
        if self.input_mode not in {"shared_tensors", "file_paths"}:
            raise ValueError("input_mode must be shared_tensors or file_paths")
        if self.handoff_mode not in {"pinned_cpu", "gpu"}:
            raise ValueError("handoff_mode must be pinned_cpu or gpu")


@dataclass
class ResplatInferenceResult:
    packet: LocalGaussianPacket
    gaussians: Any
    batch: dict[str, Any]
    inference_sec: float


class ResplatPacketGenerator:
    """Persistent in-process ReSplat model with per-frame local packet inference."""

    def __init__(self, config: ResplatRuntimeConfig) -> None:
        self.config = config
        self.repo = config.repo.expanduser().resolve()
        self.device = torch.device(config.device)
        self.model: Any = None
        self.cfg: Any = None
        self.image_shape: tuple[int, int] = (0, 0)
        self.near = 0.0
        self.far = 0.0
        self.normalize_intrinsics = True
        self.K_pixel = build_pixel_intrinsic(config.fx, config.fy, config.cx, config.cy)
        self.stream: Optional[torch.cuda.Stream] = None

    def initialize(self) -> None:
        if self.model is not None:
            return
        if not (self.repo / "config").is_dir():
            raise FileNotFoundError(f"ReSplat config directory not found: {self.repo / 'config'}")
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError(
                f"ReSplat requires CUDA; device={self.device}, "
                f"cuda_available={torch.cuda.is_available()}"
            )
        if str(self.repo) not in sys.path:
            sys.path.insert(0, str(self.repo))

        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra

        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
        overrides = [
            f"+experiment={self.config.experiment}",
            "mode=test",
            "wandb.mode=disabled",
            f"output_dir={self.config.output_dir.expanduser().resolve()}",
        ]
        overrides.extend(self.config.overrides)
        if self.config.checkpoint is not None:
            overrides.append(
                "checkpointing.pretrained_model="
                + str(self.config.checkpoint.expanduser().resolve())
            )

        with temporary_cwd(self.repo):
            with initialize_config_dir(
                config_dir=str(self.repo / "config"), version_base=None
            ):
                cfg_dict = compose(config_name="main", overrides=overrides)

            from src.config import load_typed_root_config
            from src.global_cfg import set_cfg
            from src.loss import get_losses
            from src.misc.step_tracker import StepTracker
            from src.model.decoder import get_decoder
            from src.model.encoder import get_encoder
            from src.model.model_wrapper import ModelWrapper

            cfg = load_typed_root_config(cfg_dict)
            set_cfg(cfg_dict)
            encoder, encoder_visualizer = get_encoder(cfg.model.encoder)
            model = ModelWrapper(
                cfg.optimizer,
                cfg.test,
                cfg.train,
                encoder,
                encoder_visualizer,
                get_decoder(cfg.model.decoder, cfg.dataset),
                get_losses(cfg.loss),
                StepTracker(),
                eval_data_cfg=None,
            )

            strict_load = not cfg.checkpointing.no_strict_load
            pretrained = _resolve_repo_relative(
                cfg.checkpointing.pretrained_model, self.repo
            )
            if pretrained is not None:
                checkpoint = torch.load(pretrained, map_location="cpu")
                if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                    checkpoint = checkpoint["state_dict"]
                model.load_state_dict(checkpoint, strict=strict_load)

            depth_path = _resolve_repo_relative(
                cfg.checkpointing.pretrained_depth, self.repo
            )
            if depth_path is not None:
                checkpoint = torch.load(depth_path, map_location="cpu")
                if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                    checkpoint = checkpoint["state_dict"]
                if isinstance(checkpoint, dict) and "model" in checkpoint:
                    checkpoint = checkpoint["model"]
                model.encoder.depth_predictor.load_state_dict(
                    checkpoint, strict=strict_load
                )

            update_path = _resolve_repo_relative(
                cfg.checkpointing.resume_update_module, self.repo
            )
            if update_path is not None:
                checkpoint = torch.load(update_path, map_location="cpu")
                if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                    checkpoint = checkpoint["state_dict"]
                if isinstance(checkpoint, dict) and "model" in checkpoint:
                    checkpoint = checkpoint["model"]
                model_state = model.state_dict()
                filtered = {
                    key: value
                    for key, value in checkpoint.items()
                    if "encoder.update" in key
                    and key in model_state
                    and tuple(value.shape) == tuple(model_state[key].shape)
                }
                model.load_state_dict(filtered, strict=False)

        self.cfg = cfg
        self.model = model.eval().to(self.device)
        dataset_cfg = cfg.dataset
        self.image_shape = tuple(int(value) for value in dataset_cfg.image_shape)
        self.near = float(dataset_cfg.near)
        self.far = float(dataset_cfg.far)
        self.normalize_intrinsics = bool(dataset_cfg.normalize_intrinsics)
        if self.config.refine_steps > 0:
            built_steps = int(getattr(self.model.encoder.cfg, "num_refine", 0))
            if built_steps <= 0 or not hasattr(self.model.encoder, "update_module"):
                raise RuntimeError(
                    "refinement requested but the loaded encoder has no update module"
                )
            self.model.encoder.cfg.num_refine = self.config.refine_steps
        current = torch.cuda.current_stream(self.device)
        self.stream = torch.cuda.Stream(device=self.device)
        self.stream.wait_stream(current)

    def close(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None
            torch.cuda.empty_cache()

    def _load_input_images(
        self, frame_input: StereoFrameInput
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        frame_input.validate(deep=self.config.strict_validation)
        if self.config.input_mode == "file_paths":
            left, K_left = process_image_and_intrinsic(
                frame_input.descriptor.left_path,
                self.K_pixel,
                self.image_shape,
                self.normalize_intrinsics,
            )
            right, K_right = process_image_and_intrinsic(
                frame_input.descriptor.right_path,
                self.K_pixel,
                self.image_shape,
                self.normalize_intrinsics,
            )
            return left, right, K_left, K_right

        # Preserve the validated PIL resize/center-crop behavior while avoiding a
        # second disk decode. The MAC-VO data loader has already decoded these
        # tensors. Converting the normalized tensor back to PIL is deterministic
        # for the original uint8 TartanAir source and keeps legacy interpolation.
        left_pil = TF.to_pil_image(frame_input.left_image.detach().cpu().clamp(0, 1))
        right_pil = TF.to_pil_image(frame_input.right_image.detach().cpu().clamp(0, 1))
        left, K_left = process_pil_image_and_intrinsic(
            left_pil, self.K_pixel, self.image_shape, self.normalize_intrinsics
        )
        right, K_right = process_pil_image_and_intrinsic(
            right_pil, self.K_pixel, self.image_shape, self.normalize_intrinsics
        )
        return left, right, K_left, K_right

    def _make_batch(
        self,
        frame_input: StereoFrameInput,
        T_left_c2w: torch.Tensor,
    ) -> dict[str, Any]:
        descriptor = frame_input.descriptor
        left, right, K_left, K_right = self._load_input_images(frame_input)
        T_left = T_left_c2w.detach().cpu().float()
        T_right = T_left @ fixed_tartanair_stereo_rig_cv(
            frame_input.baseline_m, dtype=torch.float32
        )
        context = {
            "extrinsics": torch.stack([T_left, T_right], dim=0).unsqueeze(0),
            "intrinsics": torch.stack([K_left, K_right], dim=0).unsqueeze(0),
            "image": torch.stack([left, right], dim=0).unsqueeze(0),
            "near": torch.full((1, 2), self.near, dtype=torch.float32),
            "far": torch.full((1, 2), self.far, dtype=torch.float32),
            "index": torch.tensor(
                [[descriptor.frame_index, descriptor.frame_index]], dtype=torch.long
            ),
            "camera_id": torch.tensor([[0, 1]], dtype=torch.long),
        }
        target = {
            "extrinsics": T_left.reshape(1, 1, 4, 4),
            "intrinsics": K_left.reshape(1, 1, 3, 3),
            "image": left.reshape(1, 1, *left.shape),
            "near": torch.full((1, 1), self.near, dtype=torch.float32),
            "far": torch.full((1, 1), self.far, dtype=torch.float32),
            "index": torch.tensor([[descriptor.frame_index]], dtype=torch.long),
            "camera_id": torch.tensor([[0]], dtype=torch.long),
        }
        return {
            "context": context,
            "target": target,
            "scene": [f"frame_{descriptor.frame_index:06d}"],
            "scene_name": ["stream"],
        }

    @torch.inference_mode()
    def infer(
        self,
        frame_input: StereoFrameInput,
        *,
        output_frame: Literal["left_camera_local", "world"] = "left_camera_local",
        T_world_from_left: Optional[torch.Tensor] = None,
        keep_gpu_packet: Optional[bool] = None,
    ) -> ResplatInferenceResult:
        self.initialize()
        assert self.model is not None
        assert self.stream is not None
        if output_frame == "left_camera_local":
            T_left = torch.eye(4, dtype=torch.float32)
        else:
            if T_world_from_left is None:
                raise ValueError("world output requires T_world_from_left")
            T_left = T_world_from_left.detach().cpu().float()

        descriptor = frame_input.descriptor
        if keep_gpu_packet is None:
            keep_gpu_packet = self.config.handoff_mode == "gpu"
        batch_cpu = self._make_batch(frame_input, T_left)

        # The host-to-device copies below are enqueued on the caller's current CUDA
        # stream. ReSplat runs on its own persistent stream, so that stream must wait
        # for the per-batch copies on every inference call. Waiting only once during
        # initialize() is insufficient and can let the encoder consume incomplete
        # input tensors, severely degrading the generated Gaussians.
        producer_stream = torch.cuda.current_stream(self.device)
        batch = _tensor_to_device(batch_cpu, self.device)
        self.stream.wait_stream(producer_stream)

        start = time.perf_counter()
        with torch.cuda.stream(self.stream):
            batch = self.model.data_shim(batch)
            encoded = self.model.encoder(
                batch["context"],
                0,
                deterministic=self.config.deterministic,
                visualization_dump=None,
            )
            if isinstance(encoded, dict):
                gaussians = encoded["gaussians"]
                condition = encoded.get("condition_features")
            else:
                gaussians = encoded
                condition = None
            if self.config.refine_steps > 0:
                if condition is None:
                    raise RuntimeError("ReSplat refinement requires condition_features")
                refined = self.model.encoder.forward_update(
                    batch["context"],
                    batch["target"] if self.config.refine_use_target else None,
                    condition,
                    gaussians,
                    self.model.decoder,
                    batch.get("context_remain"),
                )
                if len(refined["gaussian"]) < self.config.refine_steps:
                    raise RuntimeError("ReSplat returned too few refinement stages")
                gaussians = refined["gaussian"][self.config.refine_steps - 1]
        self.stream.synchronize()
        inference_sec = time.perf_counter() - start

        packet = packet_from_resplat_gaussians(
            descriptor,
            gaussians,
            batch["context"],
            inference_sec=inference_sec,
            metadata={
                "resplat_output_frame": output_frame,
                "refine_steps": self.config.refine_steps,
                "model_initialized_once": True,
                "deterministic": self.config.deterministic,
            },
            move_to_cpu=not keep_gpu_packet,
            pin_memory=self.config.pin_output_memory,
        )
        if output_frame == "world":
            # EncoderReSplat stores scales and rotations in local/scipy form even
            # though means/covariances are already transformed to world space.
            # Make the minimal packet internally consistent before handing it off.
            packet.rotations_xyzw = rotate_local_quaternions_to_world_xyzw(
                packet.rotations_xyzw, T_left.to(packet.rotations_xyzw)
            )
        packet.coordinate_frame = output_frame
        packet.metadata["input_mode"] = self.config.input_mode
        packet.metadata["handoff_mode"] = self.config.handoff_mode
        packet.validate(deep=self.config.strict_validation)
        return ResplatInferenceResult(
            packet=packet,
            gaussians=gaussians,
            batch=batch,
            inference_sec=inference_sec,
        )

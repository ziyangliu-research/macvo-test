from __future__ import annotations

import math

import torch

from .contracts import BackendUpdate


class StreamingCamera:
    """Minimal GraphDECO training/evaluation camera built from an in-memory frame."""

    def __init__(self, update: BackendUpdate, get_projection_matrix, device: torch.device):
        update.validate()
        image = update.observation.image.detach().to(
            device=device, dtype=torch.float32, non_blocking=True
        ).clamp(0.0, 1.0)
        K = update.observation.intrinsic_pixel.detach().float()
        height, width = int(image.shape[1]), int(image.shape[2])
        fx, fy = float(K[0, 0]), float(K[1, 1])
        if fx <= 0 or fy <= 0:
            raise ValueError(f"invalid camera focal lengths fx={fx}, fy={fy}")
        self.FoVx = 2.0 * math.atan(width / (2.0 * fx))
        self.FoVy = 2.0 * math.atan(height / (2.0 * fy))
        self.image_width = width
        self.image_height = height
        self.original_image = image
        self.alpha_mask = torch.ones((1, height, width), device=device)
        self.image_name = f"frame_{update.descriptor.frame_index:06d}"
        self.uid = update.descriptor.sequence_index
        self.colmap_id = self.uid
        self.data_device = device
        self.znear = 0.01
        self.zfar = 100.0

        Twc = update.pose.T_world_from_left.detach().to(
            device=device, dtype=torch.float32
        )
        Tcw = torch.linalg.inv(Twc)
        self.world_view_transform = Tcw.transpose(0, 1).contiguous()
        self.projection_matrix = get_projection_matrix(
            znear=self.znear,
            zfar=self.zfar,
            fovX=self.FoVx,
            fovY=self.FoVy,
        ).transpose(0, 1).to(device)
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0)
            .bmm(self.projection_matrix.unsqueeze(0))
            .squeeze(0)
        )
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        self.frame_index = update.descriptor.frame_index
        self.sequence_index = update.descriptor.sequence_index
        self.is_test = update.descriptor.is_test

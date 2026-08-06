from __future__ import annotations

from dataclasses import replace
from typing import Optional

import torch

from .contracts import LocalGaussianPacket


def fixed_tartanair_stereo_rig_cv(
    baseline: float = 0.25000006,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return the fixed right-camera c2w composition transform in OpenCV axes.

    It is used as:
        T_world_from_right = T_world_from_left @ T_left_from_right
    """

    if baseline <= 0:
        raise ValueError(f"baseline must be positive, got {baseline}")
    T_tartan_from_cv = torch.eye(4, dtype=dtype, device=device)
    T_tartan_from_cv[:3, :3] = torch.tensor(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=dtype,
        device=device,
    )
    T_cv_from_tartan = torch.linalg.inv(T_tartan_from_cv)
    T_rel_tartan = torch.eye(4, dtype=dtype, device=device)
    T_rel_tartan[:3, 3] = torch.tensor(
        [0.0, float(baseline), 0.0], dtype=dtype, device=device
    )
    return T_cv_from_tartan @ T_rel_tartan @ T_tartan_from_cv


def normalize_quaternion_xyzw(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return q / torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min(eps)


def quaternion_multiply_xyzw(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product for broadcastable xyzw quaternions."""

    ax, ay, az, aw = a.unbind(dim=-1)
    bx, by, bz, bw = b.unbind(dim=-1)
    return torch.stack(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        dim=-1,
    )


def quaternion_xyzw_to_matrix(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Match ReSplat's scipy-order quaternion-to-matrix convention."""

    q = normalize_quaternion_xyzw(q, eps=eps)
    x, y, z, w = q.unbind(dim=-1)
    two_s = 2.0 / ((q * q).sum(dim=-1) + eps)
    values = torch.stack(
        [
            1 - two_s * (y * y + z * z),
            two_s * (x * y - z * w),
            two_s * (x * z + y * w),
            two_s * (x * y + z * w),
            1 - two_s * (x * x + z * z),
            two_s * (y * z - x * w),
            two_s * (x * z - y * w),
            two_s * (y * z + x * w),
            1 - two_s * (x * x + y * y),
        ],
        dim=-1,
    )
    return values.reshape(*q.shape[:-1], 3, 3)


def rotation_matrix_to_quaternion_xyzw(matrix: torch.Tensor) -> torch.Tensor:
    """Convert proper rotation matrices to normalized scipy-order quaternions."""

    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"rotation matrix must end in [3,3], got {tuple(matrix.shape)}")
    m = matrix
    m00, m01, m02 = m[..., 0, 0], m[..., 0, 1], m[..., 0, 2]
    m10, m11, m12 = m[..., 1, 0], m[..., 1, 1], m[..., 1, 2]
    m20, m21, m22 = m[..., 2, 0], m[..., 2, 1], m[..., 2, 2]

    q_abs = torch.sqrt(
        torch.clamp(
            torch.stack(
                [
                    1.0 + m00 + m11 + m22,
                    1.0 + m00 - m11 - m22,
                    1.0 - m00 + m11 - m22,
                    1.0 - m00 - m11 + m22,
                ],
                dim=-1,
            ),
            min=0.0,
        )
    )
    candidates_wxyz = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )
    candidates_wxyz = candidates_wxyz / (2.0 * q_abs).clamp_min(1e-8).unsqueeze(-1)
    best = q_abs.argmax(dim=-1)
    gather_index = best[..., None, None].expand(*best.shape, 1, 4)
    wxyz = torch.gather(candidates_wxyz, -2, gather_index).squeeze(-2)
    xyzw = torch.cat([wxyz[..., 1:], wxyz[..., :1]], dim=-1)
    xyzw = normalize_quaternion_xyzw(xyzw)
    return torch.where(xyzw[..., 3:4] < 0, -xyzw, xyzw)


def quaternion_xyzw_to_wxyz(q: torch.Tensor) -> torch.Tensor:
    q = normalize_quaternion_xyzw(q)
    return torch.cat([q[..., 3:4], q[..., :3]], dim=-1)


def transform_means(T_world_from_local: torch.Tensor, means_local: torch.Tensor) -> torch.Tensor:
    if T_world_from_local.shape != (4, 4):
        raise ValueError("T_world_from_local must have shape [4,4]")
    if means_local.ndim != 2 or means_local.shape[-1] != 3:
        raise ValueError("means_local must have shape [N,3]")
    R = T_world_from_local[:3, :3].to(device=means_local.device, dtype=means_local.dtype)
    t = T_world_from_local[:3, 3].to(device=means_local.device, dtype=means_local.dtype)
    return means_local @ R.transpose(0, 1) + t


def transform_covariances(
    T_world_from_local: torch.Tensor, covariances_local: torch.Tensor
) -> torch.Tensor:
    if covariances_local.ndim != 3 or covariances_local.shape[-2:] != (3, 3):
        raise ValueError("covariances_local must have shape [N,3,3]")
    R = T_world_from_local[:3, :3].to(
        device=covariances_local.device, dtype=covariances_local.dtype
    )
    return R.unsqueeze(0) @ covariances_local @ R.transpose(0, 1).unsqueeze(0)


def rotate_local_quaternions_to_world_xyzw(
    rotations_local_xyzw: torch.Tensor,
    T_world_from_local: torch.Tensor,
) -> torch.Tensor:
    R = T_world_from_local[:3, :3].to(
        device=rotations_local_xyzw.device, dtype=rotations_local_xyzw.dtype
    )
    q_world_from_local = rotation_matrix_to_quaternion_xyzw(R).expand(
        rotations_local_xyzw.shape[0], 4
    )
    return normalize_quaternion_xyzw(
        quaternion_multiply_xyzw(q_world_from_local, rotations_local_xyzw)
    )


def rotate_harmonics_local_to_world(
    harmonics_local: torch.Tensor,
    T_world_from_local: torch.Tensor,
) -> torch.Tensor:
    """Rotate real SH coefficients from the local camera basis to world basis.

    ReSplat's current configuration uses ``no_rotate_sh=true``. Consequently a
    packet inferred with identity extrinsics carries SH coefficients in the
    canonical left-camera basis. Rigidly moving only means/covariances is not
    appearance preserving for degree > 0. This function mirrors ReSplat's own
    ``src.misc.sh_rotation.rotate_sh`` implementation using the c2w rotation.
    """

    if harmonics_local.ndim != 3 or harmonics_local.shape[-2] != 3:
        raise ValueError(
            "harmonics_local must have shape [N,3,D], got "
            f"{tuple(harmonics_local.shape)}"
        )
    coeff_count = int(harmonics_local.shape[-1])
    degree_plus_one = int(round(coeff_count**0.5))
    if degree_plus_one * degree_plus_one != coeff_count:
        raise ValueError(f"SH coefficient count must be a square, got {coeff_count}")
    if coeff_count == 1:
        return harmonics_local

    R = T_world_from_local[:3, :3].to(
        device=harmonics_local.device, dtype=harmonics_local.dtype
    )
    identity = torch.eye(3, device=R.device, dtype=R.dtype)
    if torch.allclose(R, identity, atol=1e-7, rtol=1e-7):
        return harmonics_local

    try:
        from src.misc.sh_rotation import rotate_sh
    except Exception as exc:
        raise RuntimeError(
            "Rotating ReSplat SH coefficients requires the ReSplat repository and "
            "its e3nn dependency to be importable before backend packet alignment."
        ) from exc

    # [1,1,3,3] broadcasts across Gaussian and RGB dimensions of [N,3,D].
    return rotate_sh(harmonics_local, R.reshape(1, 1, 3, 3))


def align_local_packet_to_world(
    packet: LocalGaussianPacket,
    T_world_from_left: torch.Tensor,
) -> LocalGaussianPacket:
    """Rigidly align a left-local ReSplat packet to the global map frame."""

    packet.validate()
    if packet.coordinate_frame != "left_camera_local":
        raise ValueError(f"unexpected packet frame: {packet.coordinate_frame}")
    T = T_world_from_left.to(device=packet.means.device, dtype=packet.means.dtype)
    means_world = transform_means(T, packet.means)
    rotations_world = rotate_local_quaternions_to_world_xyzw(
        packet.rotations_xyzw, T
    )
    harmonics_world = rotate_harmonics_local_to_world(packet.harmonics, T)
    context_world = T.unsqueeze(0) @ packet.context_extrinsics.to(
        device=T.device, dtype=T.dtype
    )
    metadata = dict(packet.metadata)
    metadata.update(
        {
            "aligned_from": "left_camera_local",
            "alignment_rule": (
                "means=R*x+t; covariance orientation=R*R_local; "
                "SH=WignerD(R)*SH_local; scale/opacity unchanged"
            ),
            "rotation_convention": "ReSplat xyzw",
            "sh_basis": "rotated from canonical left-camera basis to world basis",
        }
    )
    return replace(
        packet,
        means=means_world,
        rotations_xyzw=rotations_world,
        harmonics=harmonics_world,
        context_extrinsics=context_world,
        coordinate_frame="world",
        metadata=metadata,
    )

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import cv2
import numpy as np
import pypose as pp
import torch
from scipy.spatial.transform import Rotation, Slerp
from torch.utils.data import Dataset

from ..Interface import StereoData, StereoFrame
from ..SequenceBase import SequenceBase


class ETH3DRectifiedStereoSequence(SequenceBase[StereoFrame]):
    """Rectified ETH3D RGB stereo sequence produced by rectify_eth3d_stereo.py.

    Convention:
      left  = ETH3D camera 2 / rgb2, after stereo rectification
      right = ETH3D camera 1 / rgb,  after stereo rectification
      T_BS  = identity because the frame itself is the rectified-left sensor frame
      GT    = rectified-left camera c2w, interpolated to image timestamps
    """

    @classmethod
    def name(cls) -> str:
        return "ETH3D_Rectified"

    def __init__(self, config: SimpleNamespace | dict[str, Any]) -> None:
        cfg = self.config_dict2ns(config)
        self.seqRoot = Path(cfg.root).expanduser().resolve()
        calibration_path = self.seqRoot / "calibration.json"
        if not calibration_path.is_file():
            raise FileNotFoundError(f"missing ETH3D calibration: {calibration_path}")
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))

        self.ImageL = ETH3DMonocularDataset(self.seqRoot / "image_left")
        self.ImageR = ETH3DMonocularDataset(self.seqRoot / "image_right")
        if len(self.ImageL) != len(self.ImageR):
            raise ValueError(
                f"ETH3D rectified stereo count mismatch: left={len(self.ImageL)} "
                f"right={len(self.ImageR)}"
            )
        if [p.name for p in self.ImageL.file_names] != [p.name for p in self.ImageR.file_names]:
            raise ValueError("ETH3D rectified left/right filenames are not synchronized")

        timestamps_path = self.seqRoot / "timestamps.txt"
        if not timestamps_path.is_file():
            raise FileNotFoundError(f"missing ETH3D timestamps: {timestamps_path}")
        timestamp_strings = [
            line.strip() for line in timestamps_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(timestamp_strings) != len(self.ImageL):
            raise ValueError(
                f"timestamps/image count mismatch: timestamps={len(timestamp_strings)} "
                f"images={len(self.ImageL)}"
            )
        self.timestamps_sec = np.asarray([float(x) for x in timestamp_strings], dtype=np.float64)
        self.timestamps_ns = np.rint(self.timestamps_sec * 1e9).astype(np.int64)

        K_left = np.asarray(calibration["K_rectified_left"], dtype=np.float64).reshape(3, 3)
        K_right = np.asarray(calibration["K_rectified_right"], dtype=np.float64).reshape(3, 3)
        if not np.allclose(K_left, K_right, atol=1e-5, rtol=1e-6):
            raise ValueError(
                "ETH3D rectified loader requires a shared left/right K; "
                f"left={K_left.tolist()} right={K_right.tolist()}"
            )
        self.K = torch.tensor(K_left, dtype=torch.float32).unsqueeze(0)

        size = calibration["rectified_size"]
        self.width, self.height = int(size[0]), int(size[1])
        sample_l = self.ImageL[0]
        sample_r = self.ImageR[0]
        if (sample_l.shape[-1], sample_l.shape[-2]) != (self.width, self.height):
            raise ValueError(
                f"left image size {sample_l.shape[-1]}x{sample_l.shape[-2]} != "
                f"calibration {self.width}x{self.height}"
            )
        if sample_l.shape != sample_r.shape:
            raise ValueError("rectified left/right tensor shapes differ")

        baseline = calibration.get("baseline_rectified_m", calibration.get("baseline_m"))
        if baseline is None:
            raise ValueError("calibration.json has no rectified baseline")
        self.baseline = float(baseline)
        if self.baseline <= 0:
            raise ValueError(f"invalid ETH3D baseline: {self.baseline}")

        signed_baseline = calibration.get("signed_baseline_rectified_m")
        if signed_baseline is not None and float(signed_baseline) <= 0:
            raise ValueError(
                "ETH3D rectified rig has negative signed baseline for the chosen "
                "left=cam2/right=cam1 convention; check camera ordering"
            )

        self.T_BS = pp.identity_SE3(1, dtype=torch.float32)
        self.gt_pose_data: pp.LieTensor | None = None
        if bool(getattr(cfg, "gt_pose", True)):
            self.gt_pose_data = self._load_interpolated_gt(
                self.seqRoot / "groundtruth_left.txt", self.timestamps_sec
            )

        super().__init__(len(self.ImageL))

    @staticmethod
    def _load_interpolated_gt(path: Path, image_times: np.ndarray) -> pp.LieTensor:
        if not path.is_file():
            raise FileNotFoundError(f"missing ETH3D rectified-left GT: {path}")
        rows = np.loadtxt(path, dtype=np.float64)
        if rows.ndim == 1:
            rows = rows.reshape(1, -1)
        if rows.ndim != 2 or rows.shape[1] != 8:
            raise ValueError(f"ETH3D GT must be timestamp tx ty tz qx qy qz qw, got {rows.shape}")
        gt_t = rows[:, 0]
        if not np.all(np.diff(gt_t) > 0):
            raise ValueError("ETH3D GT timestamps are not strictly increasing")
        if image_times[0] < gt_t[0] or image_times[-1] > gt_t[-1]:
            raise ValueError(
                "ETH3D image timestamps fall outside GT interval: "
                f"images=[{image_times[0]}, {image_times[-1]}], "
                f"gt=[{gt_t[0]}, {gt_t[-1]}]"
            )

        translation = np.column_stack(
            [np.interp(image_times, gt_t, rows[:, axis]) for axis in range(1, 4)]
        )
        rotation = Slerp(gt_t, Rotation.from_quat(rows[:, 4:8]))(image_times).as_quat()
        pose7 = np.concatenate([translation, rotation], axis=1)
        return pp.SE3(torch.tensor(pose7, dtype=torch.float64))

    def __getitem__(self, local_index: int) -> StereoFrame:
        index = self.get_index(local_index)
        imageL = self.ImageL[index]
        imageR = self.ImageR[index]
        timestamp_ns = int(self.timestamps_ns[index])
        gt_pose = None
        if self.gt_pose_data is not None:
            gt_pose = cast(pp.LieTensor, self.gt_pose_data[index].unsqueeze(0))

        return StereoFrame(
            idx=[local_index],
            time_ns=[timestamp_ns],
            stereo=StereoData(
                T_BS=self.T_BS,
                K=self.K,
                baseline=torch.tensor([self.baseline], dtype=torch.float32),
                width=self.width,
                height=self.height,
                time_ns=[timestamp_ns],
                imageL=imageL,
                imageR=imageR,
            ),
            gt_pose=gt_pose,
        )

    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        cls._enforce_config_spec(
            config,
            {
                "root": lambda value: isinstance(value, str),
                "gt_pose": lambda value: isinstance(value, bool),
            },
            allow_excessive_cfg=True,
        )


class ETH3DMonocularDataset(Dataset):
    def __init__(self, directory: Path) -> None:
        super().__init__()
        self.directory = directory
        if not directory.is_dir():
            raise FileNotFoundError(f"ETH3D image directory not found: {directory}")
        self.file_names = sorted(directory.glob("*.png"))
        if not self.file_names:
            raise FileNotFoundError(f"no PNG images in {directory}")

    def __len__(self) -> int:
        return len(self.file_names)

    def __getitem__(self, index: int) -> torch.Tensor:
        image = cv2.imread(str(self.file_names[index]), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"failed to read {self.file_names[index]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0)
        return tensor / 255.0

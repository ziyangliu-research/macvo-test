# MAC-VO to ReSplat Packet Pipeline

This integration keeps MAC-VO and ReSplat in separate Python subprocesses while
using one Conda environment. MAC-VO first exports a metric OpenCV camera-to-world
trajectory. The existing generic pose-to-ReSplat runner in the ZipMap research
repository then generates one Gaussian packet per selected timestamp.

## Required repositories

- MAC-VO: `/home/shiyo/Desktop/MAC-VO`
- ZipMap research repository: `/home/shiyo/Desktop/ZipMap`
- ReSplat: `/home/shiyo/Desktop/Resplat`

The ZipMap checkout must contain:

```text
run_pose_resplat_metric_packet_only.py
run_zipmap_resplat_metric_packet_only.py
run_gt_resplat_fusion_api_refine_compare_v5_skipfusion_selfrender.py
```

## P000 0-10 command

```bash
cd /home/shiyo/Desktop/MAC-VO

CUDA_VISIBLE_DEVICES=0 TORCH_COMPILE_DISABLE=1 \
python run_macvo_resplat_packet_only.py \
  --macvo_repo /home/shiyo/Desktop/MAC-VO \
  --odom /home/shiyo/Desktop/MAC-VO/Config/Experiment/MACVO/MACVO_Performant.yaml \
  --data /home/shiyo/Desktop/MAC-VO/Config/Sequence/TartanAirV2_House_easy_P000.yaml \
  --zipmap_repo /home/shiyo/Desktop/ZipMap \
  --resplat_repo /home/shiyo/Desktop/Resplat \
  --left_dir /home/shiyo/Desktop/Datasets/tartanair_v2/House/Data_easy/P000/image_lcam_front \
  --right_dir /home/shiyo/Desktop/Datasets/tartanair_v2/House/Data_easy/P000/image_rcam_front \
  --work_dir /home/shiyo/Desktop/MAC-VO/outputs/macvo_stereo_resplat_P000_0_10 \
  --scene_name P000 \
  --start_index 0 \
  --end_index 10 \
  --stride 1 \
  --stereo_baseline 0.25000006 \
  --resplat_experiment tartanair_p000_ft \
  --resplat_packet_stage init \
  --refine_steps 0 \
  --refine_use_target false \
  --resplat_target_camera left \
  --resplat_target_offset 0 \
  --packet_out_name packets \
  --fx 320 --fy 320 --cx 320 --cy 320 \
  --device cuda:0 \
  --timing
```

## Expected output

```text
macvo_stereo_resplat_P000_0_10/
├── macvo_pose/
│   ├── macvo_pose_results.npz
│   ├── trajectory_c2w_opencv.txt
│   ├── evaluation.json
│   ├── summary.json
│   └── macvo_runtime/
├── metric_pose/
├── packets/
│   ├── manifest.json
│   ├── packet_timing.csv
│   └── refine_0/
│       ├── P000_0000.pt
│       ├── ...
│       └── P000_0009.pt
├── run_summary.json
└── combined_pipeline_summary.json
```

## Pose evaluation

When the MAC-VO data config uses `gtPose: true`, MAC-VO writes `ref_poses.npy`.
The exporter matches estimated and reference poses by timestamp and reports:

- ATE RMSE after SE(3) alignment
- ATE RMSE after Sim(3) alignment
- mean, median, standard deviation, minimum, and maximum ATE
- the Sim(3) alignment scale

The terminal prints:

```text
[ATE] SE(3) RMSE=... m | Sim(3) RMSE=... m
```

The full result is stored in:

```text
macvo_pose/evaluation.json
macvo_pose/summary.json
combined_pipeline_summary.json
```

Use `--skip_ate` to disable this evaluation.

## Pose contract

`macvo_pose_results.npz` contains:

- `T_raw_accumulated_c2w_opencv`: `[N,4,4]`, metric OpenCV c2w, first frame identity
- `T_macvo_c2w_tartanair`: original MAC-VO c2w matrices
- `selected_original_indices`: original dataset frame indices
- `timestamps_ns`
- `need_interp`
- `valid`

MAC-VO is stereo and already metric, so no additional trajectory scale is applied
before ReSplat.

## Reusing an existing MAC-VO trajectory

After a successful MAC-VO stage, rerun pose export, ATE evaluation, and packet
generation by adding:

```text
--reuse_macvo_pose
```

The exporter reuses the newest `poses.npy` below the current work directory.

## Warning handling

By default, the pipeline suppresses only these known benign compatibility warnings:

- jaxtyping instrumentation warnings for nested output classes
- Hydra's missing `_self_` composition warning
- torchvision's deprecated `pretrained`/legacy `weights` warnings

These warnings do not change the current inference result. To display them again,
add:

```text
--show_known_warnings
```

All other warnings and errors remain visible.

## Current limitations

- `--stride` must be 1 because the native MAC-VO sequence runner clips a contiguous range.
- Packet generation uses MAC-VO's official saved trajectory after termination and
  post-processing. `need_interp` is recorded for diagnosis.
- The pipeline generates packets only. Fusion and incremental Gaussian optimization
  remain separate backend stages.

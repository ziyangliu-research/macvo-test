# MAC-VO to ReSplat Packet Pipeline

This integration keeps MAC-VO and ReSplat in separate Python subprocesses while
using one Conda environment. MAC-VO first exports a metric OpenCV camera-to-world
trajectory. The generic pose-to-ReSplat runner then generates one Gaussian packet
per selected timestamp. An optional third stage prepares the strict train/test
camera scene required by the incremental 3DGS backend.

## Recommended public entrypoint

Use the configuration-driven launcher instead of manually maintaining a long
command:

```bash
conda activate macvo_resplat
cd /home/shiyo/Desktop/MAC-VO

python run_pipeline_from_config.py
```

The default configuration is:

```text
Config/Pipeline/MACVO_ReSplat_P000_0_50.yaml
```

It reproduces the P000 `[0,50)` experiment and prepares:

```text
MAC-VO trajectory
ReSplat refine_0 packets
strict train/test camera scene
backend_input_manifest.json
```

Before the first full run, validate all paths without running either neural
network:

```bash
python run_pipeline_from_config.py --dry_run
```

The launcher saves:

```text
outputs/macvo_stereo_resplat_P000_0_50/
├── resolved_pipeline_config.json
└── resolved_pipeline_command.sh
```

These files record the fully resolved machine paths and the exact low-level
command used for the experiment.

## Small experiment overrides

Do not copy and edit the whole command for a new frame range. Override only the
changed YAML values:

```bash
python run_pipeline_from_config.py \
  --set sequence.end_index=100 \
  --set paths.work_dir=outputs/macvo_stereo_resplat_P000_0_100
```

Reuse an existing MAC-VO trajectory:

```bash
python run_pipeline_from_config.py \
  --set runtime.reuse_macvo_pose=true
```

Disable camera-scene preparation and generate packets only:

```bash
python run_pipeline_from_config.py \
  --set split.prepare_camera_scene=false
```

## Repository layout assumptions

The public P000 config uses repository-relative paths:

```text
Desktop/
├── MAC-VO/
├── ZipMap/
└── Resplat/
```

The dataset root is read from:

```text
Config/Sequence/TartanAirV2_House_easy_P000.yaml
```

The left/right image directories are derived automatically from that root, so
they are not duplicated in the pipeline configuration.

The current generic pose-to-ReSplat implementation is still read from the
ZipMap research repository:

```text
run_pose_resplat_metric_packet_only.py
run_zipmap_resplat_metric_packet_only.py
run_gt_resplat_fusion_api_refine_compare_v5_skipfusion_selfrender.py
```

This is a temporary integration dependency. Before the final public release,
the generic ReSplat packet generator should be extracted into a standalone
module so that the MAC-VO system no longer depends on a repository named
`ZipMap`.

## Camera-scene preparation script

The configuration expects the script at:

```text
/home/shiyo/Desktop/MAC-VO/prepare_zipmap_packet_camera_scene_only.py
```

Repository-relative form:

```text
prepare_zipmap_packet_camera_scene_only.py
```

This script must be committed to `macvo-test` before the integration branch can
be considered self-contained and ready for public release.

## Low-level entrypoint

`run_macvo_resplat_packet_only.py` remains available as an advanced interface.
It exposes all individual parameters for ablations and debugging. The config
launcher is the recommended interface for routine experiments and public
reproduction.

## Expected output

```text
macvo_stereo_resplat_P000_0_50/
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
│       └── P000_0049.pt
├── 3dgs_camera_scene_macvo_strict_split/
│   ├── images/
│   ├── transforms_train.json
│   ├── transforms_test.json
│   └── points3d.ply
├── backend_input_manifest.json
├── resolved_pipeline_config.json
├── resolved_pipeline_command.sh
├── run_summary.json
└── combined_pipeline_summary.json
```

## Pose contract

`macvo_pose_results.npz` contains:

- `T_raw_accumulated_c2w_opencv`: `[N,4,4]`, metric OpenCV c2w, first frame identity
- `T_macvo_c2w_tartanair`: original MAC-VO c2w matrices
- `selected_original_indices`: original dataset frame indices
- `timestamps_ns`
- `need_interp`
- `valid`

MAC-VO is stereo and already metric, so no additional trajectory scale is
applied before ReSplat.

When `ref_poses.npy` is available, `evaluation.json` and the console report ATE
RMSE after SE(3) and Sim(3) alignment.

## Warning policy

Known non-fatal compatibility warnings from jaxtyping, Hydra, and the deprecated
torchvision `pretrained` argument are filtered by default. Other warnings and
all errors remain visible. Set:

```bash
python run_pipeline_from_config.py \
  --set runtime.show_known_warnings=true
```

to restore the known warnings.

## Current limitations

- `stride` must be 1 because the native MAC-VO sequence runner clips a contiguous range.
- Packet generation uses MAC-VO's official saved trajectory after termination and post-processing.
- `need_interp` is recorded for diagnosis.
- The pipeline prepares backend inputs but does not start incremental Gaussian optimization.
- The pose-to-ReSplat implementation still has a temporary dependency on the ZipMap research repository.

# Asynchronous MAC-VO + ReSplat + Incremental 3DGS

This branch replaces command-to-command subprocess chaining with persistent
Python components and bounded in-memory queues. It keeps the current sibling
repository layout:

```text
Desktop/
├── MAC-VO/               # this repository and system entrypoint
├── Resplat/              # ReSplat model code/checkpoints
└── gaussian-splatting/   # GraphDECO renderer/model
```

The existing serial scripts remain untouched. The asynchronous implementation is
isolated on `feature/async-macvo-resplat-3dgs-v1` until the validation gates pass.

## Architecture

```text
                         ┌──────────────────────────────┐
Decoded StereoFrame ─────┤ MAC-VO pose worker           │
       │                 │ committed metric Twc (t-1)  │
       │                 └──────────────┬───────────────┘
       │                                │
       │                 ┌──────────────▼───────────────┐
       └────────────────►│ ReSplat packet worker        │
                         │ [I, fixed stereo baseline]    │
                         │ left-camera-local Gaussians   │
                         └──────────────┬───────────────┘
                                        │
                              ordered pose/packet join
                                        │
                         ┌──────────────▼───────────────┐
                         │ Incremental GraphDECO backend │
                         │ local packet -> world frame   │
                         │ reset / append / recent-K opt │
                         └──────────────────────────────┘
```

Three long-lived worker threads own the three GPU components. Initialization is
coordinated once, sequentially, before streaming begins. Each component can use a
dedicated CUDA stream. Input queues are bounded to provide backpressure and cap
packet memory.

## Coordinate contract

ReSplat receives a canonical stereo rig:

```text
left  c2w = I
right c2w = fixed 0.25000006 m stereo transform
```

It produces a packet in the current left-camera coordinate frame. Once MAC-VO
commits `T_world_from_left`, the backend applies:

```text
mean_world = R_world_left @ mean_local + t_world_left
cov_world  = R_world_left @ cov_local @ R_world_left^T
```

The implementation stores ReSplat rotations in the model's actual `xyzw`
convention, composes them in `xyzw`, and only then converts to GraphDECO's `wxyz`
convention. Scale and opacity are invariant to a rigid transform. SH coefficients
are intentionally left unchanged because the current `EncoderReSplat` direct-world
path also leaves them unchanged.

This contract is plausible from the implementation but **must be validated on the
actual checkpoint and sequence** before using asynchronous results in an experiment.

## Required validation gate

First generate or reuse the current serial MAC-VO pose file, then compare local
inference + alignment against the existing direct-world ReSplat path:

```bash
cd /home/shiyo/Desktop/MAC-VO

CUDA_VISIBLE_DEVICES=0 TORCH_COMPILE_DISABLE=1 \
python validate_resplat_local_world_equivalence.py \
  --resplat_repo /home/shiyo/Desktop/Resplat \
  --dataset_root /home/shiyo/Desktop/Datasets/tartanair_v2/House/Data_easy/P000 \
  --pose_npz /home/shiyo/Desktop/MAC-VO/outputs/macvo_stereo_resplat_P000_0_50/macvo_pose/macvo_pose_results.npz \
  --pose_indices 1,10,20,30,40,49 \
  --input_mode file_paths \
  --output outputs/local_world_equivalence_file_paths.json
```

Then repeat with the shared decoded-frame input used by the async system:

```bash
CUDA_VISIBLE_DEVICES=0 TORCH_COMPILE_DISABLE=1 \
python validate_resplat_local_world_equivalence.py \
  --resplat_repo /home/shiyo/Desktop/Resplat \
  --dataset_root /home/shiyo/Desktop/Datasets/tartanair_v2/House/Data_easy/P000 \
  --pose_npz /home/shiyo/Desktop/MAC-VO/outputs/macvo_stereo_resplat_P000_0_50/macvo_pose/macvo_pose_results.npz \
  --pose_indices 1,10,20,30,40,49 \
  --input_mode shared_tensors \
  --output outputs/local_world_equivalence_shared_tensors.json
```

Do not continue to the full backend if either report contains `"passed": false`.

## Dry run

```bash
CUDA_VISIBLE_DEVICES=0 TORCH_COMPILE_DISABLE=1 \
python run_async_pipeline.py \
  --config Config/Pipeline/MACVO_ReSplat_Async_P000_0_50.yaml \
  --dry_run
```

## Full experiment configuration

This retains per-packet evaluation, W&B, maintenance at local iteration 50, and
100 local iterations per train packet:

```bash
CUDA_VISIBLE_DEVICES=0 TORCH_COMPILE_DISABLE=1 \
python run_async_pipeline.py \
  --config Config/Pipeline/MACVO_ReSplat_Async_P000_0_50.yaml
```

## Throughput configuration

This disables per-packet evaluation, W&B, periodic PLY writes, and JSON updates.
The mapping algorithm remains packet reset + insertion + recent-K optimization +
maintenance:

```bash
CUDA_VISIBLE_DEVICES=0 TORCH_COMPILE_DISABLE=1 \
python run_async_pipeline.py \
  --config Config/Pipeline/MACVO_ReSplat_Async_Fast_P000_0_50.yaml
```

## Important runtime decisions

- **No required intermediate hand-off files.** NPZ, PT, JSON, and PLY are optional
  artifacts, not dependencies between stages.
- **Single decoded stereo input.** MAC-VO's loader decodes each pair once. ReSplat
  reuses the decoded tensors and preserves the legacy PIL resize/crop operation.
- **One-frame pose commitment delay.** Under the current `AllKeyframe` and
  `TwoFrame_PGO` configuration, frame `t-1` is emitted after frame `t` lets MAC-VO
  write the preceding optimizer result back to the map.
- **Strict split.** Test frames are never submitted to ReSplat and never inserted.
  Their images and committed poses are retained only for evaluation.
- **Invalid online poses.** A frame marked `need_interp` is skipped by default.
  End-of-sequence interpolation from the serial exporter cannot be used causally.
- **GPU packet hand-off.** The default configuration keeps the packet on GPU to
  avoid GPU→CPU→GPU copies. Set `resplat_frontend.handoff_mode=pinned_cpu` if VRAM
  pressure or allocator behavior is problematic.
- **Explicit online spatial scale.** The previous GraphDECO `Scene` derived a
  spatial learning-rate scale from the complete camera trajectory. An online
  system cannot see future cameras, so `backend.spatial_lr_scale` is explicit and
  must be checked against the serial baseline.

## Validation sequence after the coordinate gate

1. Run `runtime.backend_mode=null` over 0-50 to validate ordering and strict split.
2. Run the full backend with `resplat_frontend.input_mode=file_paths`; compare final
   metrics and per-packet curves with the current serial system.
3. Switch to `shared_tensors`; isolate any preprocessing difference.
4. Compare `pinned_cpu` and `gpu` hand-off for memory and throughput.
5. Compare dedicated CUDA streams on/off. One-GPU concurrency is beneficial only
   if it reduces measured end-to-end wall time without causing OOM or kernel
   contention.
6. Ablate `backend.spatial_lr_scale` if the async backend diverges from the serial
   optimization trajectory.

## Current status

The code is compile-checked and the CPU tests cover ordered joining, strict split,
SE(3) covariance alignment, and ReSplat-to-GraphDECO quaternion conventions. It
has not been executed with the three CUDA repositories in this environment.

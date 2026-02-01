# DROID Action-Conditioned World Model — Run Commands

## Setup

```bash
# Core dependencies
pip install torch torchvision numpy opencv-python tqdm h5py

# For RLDS format (Google Cloud DROID)
pip install tensorflow tensorflow_datasets

# For HuggingFace LeRobot format
pip install datasets

# For V-JEPA 2 HuggingFace fallback
pip install transformers
```

## Training Commands

### 1. Local HDF5 Files (recommended for custom setups)

```bash
# Full training — 62 hours, ViT-L backbone, default hyperparameters
python train_action_conditioned_world_model.py \
    --data_path ./data/droid_hdf5 \
    --format hdf5 \
    --vjepa2_size vitl \
    --num_epochs 50 \
    --batch_size 4 \
    --learning_rate 3e-4 \
    --encoder_lr 1e-5 \
    --predict_horizon 4 \
    --save_dir ./checkpoints/world_model
```

### 2. RLDS Format (Google Cloud Storage)

```bash
# From Google Cloud (primary DROID source)
python train_action_conditioned_world_model.py \
    --data_path gs://gresearch/robotics/droid \
    --format rlds \
    --vjepa2_size vitl \
    --num_epochs 50 \
    --batch_size 4

# From local RLDS copy
python train_action_conditioned_world_model.py \
    --data_path /path/to/local/droid_rlds \
    --format rlds \
    --vjepa2_size vitl \
    --num_epochs 50 \
    --batch_size 4
```

### 3. HuggingFace LeRobot Format

```bash
python train_action_conditioned_world_model.py \
    --data_path cadene/droid \
    --format huggingface \
    --vjepa2_size vitl \
    --num_epochs 50 \
    --batch_size 4
```

### 4. Debug / Quick Test (100 episodes, 2 epochs)

```bash
python train_action_conditioned_world_model.py \
    --data_path ./data/droid_hdf5 \
    --format hdf5 \
    --max_episodes 100 \
    --num_epochs 2 \
    --batch_size 2 \
    --vjepa2_size vitl \
    --save_dir ./checkpoints/debug
```

---

## V-JEPA 2 Backbone Options

| Flag              | Params | Embed dim | Depth | Heads | Notes                    |
|-------------------|--------|-----------|-------|-------|--------------------------|
| `--vjepa2_size vitl` | 300M   | 1024      | 24    | 16    | **Recommended** — best speed/quality |
| `--vjepa2_size vith` | 600M   | 1280      | 32    | 16    | Higher quality, 2× slower   |
| `--vjepa2_size vitg` | 1B     | 1024      | 40    | 16    | Best quality, needs ≥40GB VRAM |

---

## Multi-GPU Training

```bash
# Using PyTorch DDP (wrap with torchrun)
torchrun --nproc_per_node=4 train_action_conditioned_world_model.py \
    --data_path ./data/droid_hdf5 \
    --format hdf5 \
    --vjepa2_size vitl \
    --batch_size 4 \
    --num_epochs 50

# Using accelerate (HuggingFace)
accelerate launch --num_processes 4 train_action_conditioned_world_model.py \
    --data_path ./data/droid_hdf5 \
    --format hdf5 \
    --vjepa2_size vitl \
    --batch_size 4
```

> **Note:** The current script is single-GPU. For DDP, you'd wrap the model
> with `DistributedDataParallel` and use `DistributedSampler`.
> The commands above show the launcher pattern — actual DDP integration
> would require minor code changes.

---

## Memory Optimization

```bash
# Low VRAM (< 16 GB) — gradient checkpointing + smaller batch
python train_action_conditioned_world_model.py \
    --data_path ./data/droid_hdf5 \
    --format hdf5 \
    --vjepa2_size vitl \
    --batch_size 1 \
    --use_grad_checkpoint \
    --use_amp \
    --num_workers 2

# Medium VRAM (24 GB) — AMP + moderate batch
python train_action_conditioned_world_model.py \
    --data_path ./data/droid_hdf5 \
    --format hdf5 \
    --vjepa2_size vitl \
    --batch_size 4 \
    --use_amp \
    --num_workers 4

# High VRAM (40-80 GB) — ViT-H, larger batch, no AMP needed
python train_action_conditioned_world_model.py \
    --data_path ./data/droid_hdf5 \
    --format hdf5 \
    --vjepa2_size vith \
    --batch_size 8 \
    --num_workers 8
```

---

## Different Camera Views

```bash
# Exterior camera 1 (default — wide angle)
python train_action_conditioned_world_model.py \
    --data_path ./data/droid_hdf5 \
    --format hdf5 \
    --camera exterior_image_1_left

# Exterior camera 2 (second wide angle)
python train_action_conditioned_world_model.py \
    --data_path ./data/droid_hdf5 \
    --format hdf5 \
    --camera exterior_image_2_left

# Wrist camera (close-up, best for grasping)
python train_action_conditioned_world_model.py \
    --data_path ./data/droid_hdf5 \
    --format hdf5 \
    --camera wrist_image_left
```

---

## Resume Training from Checkpoint

```bash
python train_action_conditioned_world_model.py \
    --data_path ./data/droid_hdf5 \
    --format hdf5 \
    --pretrained_checkpoint ./checkpoints/world_model/epoch_025.pth \
    --num_epochs 50 \
    --batch_size 4
```

---

## Loss Weight Tuning

```bash
# Default weights (balanced)
python train_action_conditioned_world_model.py \
    --data_path ./data/droid_hdf5 \
    --format hdf5 \
    --lambda_latent 1.0 \
    --lambda_action 1.0 \
    --lambda_state 0.5 \
    --lambda_task 0.1 \
    --lambda_uncertainty 0.01

# Emphasis on action prediction (for policy learning downstream)
python train_action_conditioned_world_model.py \
    --data_path ./data/droid_hdf5 \
    --format hdf5 \
    --lambda_latent 0.5 \
    --lambda_action 2.0 \
    --lambda_state 1.0 \
    --lambda_task 0.1

# Emphasis on latent prediction (for representation quality)
python train_action_conditioned_world_model.py \
    --data_path ./data/droid_hdf5 \
    --format hdf5 \
    --lambda_latent 2.0 \
    --lambda_action 0.5 \
    --lambda_state 0.5 \
    --lambda_task 0.05
```

---

## All Arguments Reference

```
Data:
  --data_path              Path to DROID data (required)
  --format                 rlds | huggingface | hdf5
  --camera                 Camera view name
  --max_episodes           Cap episode count (debugging)
  --augmentation_strength  light | medium | heavy

V-JEPA 2 Backbone:
  --vjepa2_size            vitl | vith | vitg
  --vjepa2_resolution      256 | 384
  --pretrained_checkpoint  Resume from saved checkpoint

Model:
  --img_size               Input resolution (default: 224)
  --num_frames             Frames per clip (default: 16)
  --predictor_dim          Predictor hidden dim (default: 1024)
  --predictor_depth        Predictor layers (default: 12)
  --predictor_heads        Predictor attention heads (default: 8)
  --action_dim             Action dimensions (default: 7)
  --state_dim              State dimensions (default: 14)
  --num_task_classes        Task classes (default: 2)
  --predict_horizon        Future steps to predict (default: 4)

Training:
  --num_epochs             Training epochs (default: 50)
  --batch_size             Batch size (default: 4)
  --learning_rate          Predictor LR (default: 3e-4)
  --encoder_lr             Encoder LR (default: 1e-5)
  --weight_decay           AdamW weight decay (default: 0.05)
  --warmup_ratio           LR warmup fraction (default: 0.05)
  --grad_clip              Gradient clipping norm (default: 1.0)
  --ema_decay              EMA encoder decay (default: 0.999)
  --num_workers            DataLoader workers (default: 4)
  --use_amp                Enable fp16 mixed precision
  --use_grad_checkpoint    Enable gradient checkpointing

Loss Weights:
  --lambda_latent          Latent prediction weight (default: 1.0)
  --lambda_action          Action prediction weight (default: 1.0)
  --lambda_state           State prediction weight (default: 0.5)
  --lambda_task            Task classification weight (default: 0.1)
  --lambda_uncertainty     Uncertainty regularization (default: 0.01)

I/O:
  --save_dir               Checkpoint directory
  --save_every_steps       Periodic save interval (default: 5000)
  --device                 auto | cuda | mps | cpu
  --seed                   Random seed (default: 42)
```

---

## Key Design Decisions

| Aspect | Choice | Rationale |
|--------|--------|-----------|
| Min episode length | 4 seconds (60 frames) | V-JEPA 2 paper protocol |
| Data target | 62 hours | V-JEPA 2 paper used this subset |
| Encoder LR | 1e-5 (10× lower) | Pretrained backbone needs gentle fine-tuning |
| Predictor LR | 3e-4 | Standard for randomly-initialized transformer |
| EMA decay | 0.999 | Stable latent targets (V-JEPA / BYOL standard) |
| Warmup | 5% of steps | Prevents early training instability |
| Predict horizon | 4 steps | ~0.5s at tubelet resolution = 8 tubelet steps from 16 frames |
| Temporal scale | 1.0 | Manipulation is faster than surgery (which used 0.6) |

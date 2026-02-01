"""
TRAIN ACTION-CONDITIONED WORLD MODEL ON DROID (62 hours)
=========================================================

Trains the full action-conditioned world model (encoder + predictor +
autoregressive head + task classifier) on the DROID dataset, following
the V-JEPA 2 paper's protocol:

  - Downloads V-JEPA 2 pretrained ViT backbone as encoder initialization
  - Properly maps HuggingFace / native V-JEPA 2 keys to VideoViTEncoder
  - Uses EMA target encoder for latent prediction targets (V-JEPA style)
  - Skips episodes shorter than 4 seconds (< 60 frames at 15 Hz)
  - Trains on ~62 hours of manipulation data
  - Action dim = 7 (6D Cartesian target + 1 gripper)
  - State dim = 14 (7 joint + 6 Cartesian + 1 gripper)

Training Objectives:
  1. Latent prediction loss (V-JEPA style) — predict future encoder features
     from past observations + actions, using EMA target encoder targets
  2. Action prediction loss — predict future actions from predictor output
  3. State prediction loss  — predict future robot states
  4. Task classification loss — predict manipulation success / failure

Usage:
    # From HuggingFace LeRobot (recommended — no auth needed):
    python train_action_conditioned_world_model.py \
        --data_path cadene/droid \
        --format huggingface

    # From local HDF5 files:
    python train_action_conditioned_world_model.py \
        --data_path ./data/droid_hdf5 \
        --format hdf5

    # From RLDS (requires GCS auth OR local copy):
    python train_action_conditioned_world_model.py \
        --data_path /path/to/local/droid_rlds \
        --format rlds

    # Quick debug (100 episodes):
    python train_action_conditioned_world_model.py \
        --data_path ./data/droid_hdf5 \
        --format hdf5 \
        --max_episodes 100 \
        --num_epochs 2 \
        --batch_size 2

Requirements:
    pip install torch torchvision numpy opencv-python tqdm h5py
    # For HF:     pip install datasets transformers
    # For RLDS:   pip install tensorflow tensorflow_datasets
"""

import math
import copy
import json
import re
import argparse
import os
from pathlib import Path
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import cv2
from tqdm import tqdm

from droid_world_model import DROIDActionConditionedWorldModel, VideoViTEncoder


# ============================================================================
# CONSTANTS — DROID Dataset
# ============================================================================

DROID_FPS = 15                          # DROID capture rate
MIN_EPISODE_SECONDS = 4                 # V-JEPA 2 paper: skip < 4s
MIN_EPISODE_FRAMES = MIN_EPISODE_SECONDS * DROID_FPS  # = 60 frames
TARGET_HOURS = 62                       # V-JEPA 2 used 62 h of DROID
TARGET_FRAMES = int(TARGET_HOURS * 3600 * DROID_FPS)  # ~ 3.35 M frames

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ============================================================================
# V-JEPA 2 KEY MAPPING — HuggingFace / Native -> VideoViTEncoder
# ============================================================================

def map_vjepa2_state_dict(source_sd, target_encoder, embed_dim):
    """
    Map V-JEPA 2 pretrained weights to VideoViTEncoder.

    Handles TWO naming conventions automatically:

    Convention A - HuggingFace ViT (what AutoModel.from_pretrained gives):
        embeddings.patch_embeddings.proj.weight  -> tubelet_embed.projection.weight
        layer.{i}.attention.query.weight         -> blocks.{i}.attn.qkv_proj.weight (fused)
        layer.{i}.attention.key.weight           /
        layer.{i}.attention.value.weight         /
        layer.{i}.attention.output.dense.weight  -> blocks.{i}.attn.out_proj.weight
        layer.{i}.intermediate.dense.weight      -> blocks.{i}.mlp.0.weight
        layer.{i}.output.dense.weight            -> blocks.{i}.mlp.3.weight
        layernorm.weight                         -> norm.weight

    Convention B - Native V-JEPA / timm:
        patch_embed.proj.weight                  -> tubelet_embed.projection.weight
        blocks.{i}.attn.qkv.weight              -> blocks.{i}.attn.qkv_proj.weight
        blocks.{i}.attn.proj.weight              -> blocks.{i}.attn.out_proj.weight
        blocks.{i}.mlp.fc1.weight               -> blocks.{i}.mlp.0.weight
        blocks.{i}.mlp.fc2.weight               -> blocks.{i}.mlp.3.weight
        norm.weight                              -> norm.weight

    Also handles:
      - Separate Q/K/V -> fused QKV concatenation
      - 2D Conv (4D tensor) -> 3D Conv (5D tensor) inflation
      - HF biases -> skipped where target has bias=False
      - Depth mismatch -> loads min(source, target) layers
      - RoPE buffers -> skipped (computed from config)
    """
    target_sd = target_encoder.state_dict()
    mapped = {}

    all_keys = set(source_sd.keys())
    is_hf = any('embeddings.patch_embeddings' in k for k in all_keys)
    has_separate_qkv = any('.attention.query.' in k or '.attention.key.' in k for k in all_keys)
    has_fused_qkv = any('.attn.qkv.' in k for k in all_keys)

    if is_hf:
        convention = "HuggingFace ViT"
    elif has_fused_qkv:
        convention = "Native V-JEPA / timm"
    else:
        convention = "Unknown (best-effort)"
    print(f"    Detected convention: {convention}")

    # Detect source depth
    layer_indices = set()
    for k in all_keys:
        m = re.search(r'(?:layer|blocks)\.(\d+)\.', k)
        if m:
            layer_indices.add(int(m.group(1)))
    source_depth = max(layer_indices) + 1 if layer_indices else 0
    target_depth = len(target_encoder.blocks)
    load_depth = min(source_depth, target_depth)
    if source_depth != target_depth:
        print(f"    Depth mismatch: source={source_depth}, target={target_depth}, "
              f"loading {load_depth}")

    used_source_keys = set()
    skipped_biases = []

    def find_key_exact(*candidates):
        for c in candidates:
            if c in all_keys:
                return c
        return None

    # ==================================================================
    # 1. PATCH EMBEDDING -> tubelet_embed.projection
    # ==================================================================
    patch_w_key = find_key_exact(
        'embeddings.patch_embeddings.proj.weight',
        'patch_embed.proj.weight',
        'embeddings.projection.weight',
    )
    patch_b_key = find_key_exact(
        'embeddings.patch_embeddings.proj.bias',
        'patch_embed.proj.bias',
        'embeddings.projection.bias',
    )

    if patch_w_key:
        src_w = source_sd[patch_w_key]
        used_source_keys.add(patch_w_key)
        target_shape = target_sd['tubelet_embed.projection.weight'].shape

        if src_w.dim() == 5:
            if src_w.shape == target_shape:
                mapped['tubelet_embed.projection.weight'] = src_w
                print(f"    Patch embed: 3D direct copy {src_w.shape}")
            else:
                out_ch, in_ch = src_w.shape[:2]
                t_target = target_shape[2]
                w_3d = torch.zeros(target_shape, dtype=src_w.dtype)
                t_src = src_w.shape[2]
                t_copy = min(t_src, t_target)
                src_start = max(0, (t_src - t_copy) // 2)
                tgt_start = max(0, (t_target - t_copy) // 2)
                w_3d[:, :, tgt_start:tgt_start+t_copy] = src_w[:, :, src_start:src_start+t_copy]
                mapped['tubelet_embed.projection.weight'] = w_3d
                print(f"    Patch embed: adapted {src_w.shape} -> {w_3d.shape}")
        elif src_w.dim() == 4:
            out_ch, in_ch, h, w = src_w.shape
            t_target = target_shape[2]
            w_3d = torch.zeros(out_ch, in_ch, t_target, h, w, dtype=src_w.dtype)
            center = t_target // 2
            w_3d[:, :, center, :, :] = src_w
            mapped['tubelet_embed.projection.weight'] = w_3d
            print(f"    Patch embed: inflated 2D {src_w.shape} -> 3D {w_3d.shape}")
        else:
            print(f"    Patch embed: unexpected dim={src_w.dim()}, skipping")

    if patch_b_key and 'tubelet_embed.projection.bias' in target_sd:
        mapped['tubelet_embed.projection.bias'] = source_sd[patch_b_key]
        used_source_keys.add(patch_b_key)

    # ==================================================================
    # 2. TRANSFORMER BLOCKS
    # ==================================================================
    for i in range(load_depth):
        # Find source layer prefix
        if f'layer.{i}.norm1.weight' in all_keys:
            src_pfx = f'layer.{i}'
        elif f'blocks.{i}.norm1.weight' in all_keys:
            src_pfx = f'blocks.{i}'
        elif f'encoder.layer.{i}.norm1.weight' in all_keys:
            src_pfx = f'encoder.layer.{i}'
        else:
            print(f"    Layer {i}: not found in source, skipping")
            continue

        tgt_pfx = f'blocks.{i}'

        # -- Norm 1 --
        for sfx in ['weight', 'bias']:
            src_k = find_key_exact(
                f'{src_pfx}.norm1.{sfx}',
                f'{src_pfx}.layernorm_before.{sfx}',
                f'{src_pfx}.attention.output.LayerNorm.{sfx}',
            )
            tgt_k = f'{tgt_pfx}.norm1.{sfx}'
            if src_k and tgt_k in target_sd:
                mapped[tgt_k] = source_sd[src_k]
                used_source_keys.add(src_k)

        # -- Attention Q/K/V -> fused QKV --
        tgt_qkv_key = f'{tgt_pfx}.attn.qkv_proj.weight'

        if has_separate_qkv:
            q_key = find_key_exact(
                f'{src_pfx}.attention.query.weight',
                f'{src_pfx}.attention.attention.query.weight',
                f'{src_pfx}.attn.q_proj.weight',
                f'{src_pfx}.self_attn.q_proj.weight',
            )
            k_key = find_key_exact(
                f'{src_pfx}.attention.key.weight',
                f'{src_pfx}.attention.attention.key.weight',
                f'{src_pfx}.attn.k_proj.weight',
                f'{src_pfx}.self_attn.k_proj.weight',
            )
            v_key = find_key_exact(
                f'{src_pfx}.attention.value.weight',
                f'{src_pfx}.attention.attention.value.weight',
                f'{src_pfx}.attn.v_proj.weight',
                f'{src_pfx}.self_attn.v_proj.weight',
            )

            if q_key and k_key and v_key:
                q_w, k_w, v_w = source_sd[q_key], source_sd[k_key], source_sd[v_key]
                qkv_w = torch.cat([q_w, k_w, v_w], dim=0)
                mapped[tgt_qkv_key] = qkv_w
                used_source_keys.update([q_key, k_key, v_key])
                if i == 0:
                    print(f"    Attention: fused Q{q_w.shape}+K+V -> QKV{qkv_w.shape}")
                # Skip biases (target has bias=False)
                for bk in [q_key.replace('.weight', '.bias'),
                           k_key.replace('.weight', '.bias'),
                           v_key.replace('.weight', '.bias')]:
                    if bk in all_keys:
                        used_source_keys.add(bk)
                        skipped_biases.append(bk)

        elif has_fused_qkv:
            qkv_key = find_key_exact(
                f'{src_pfx}.attn.qkv.weight',
                f'{src_pfx}.attention.qkv.weight',
            )
            if qkv_key:
                mapped[tgt_qkv_key] = source_sd[qkv_key]
                used_source_keys.add(qkv_key)
                qkv_b = qkv_key.replace('.weight', '.bias')
                if qkv_b in all_keys:
                    used_source_keys.add(qkv_b)
                    skipped_biases.append(qkv_b)
                if i == 0:
                    print(f"    Attention: direct QKV copy {source_sd[qkv_key].shape}")

        # -- Attention output projection --
        tgt_out_key = f'{tgt_pfx}.attn.out_proj.weight'
        out_src = find_key_exact(
            f'{src_pfx}.attention.output.dense.weight',
            f'{src_pfx}.attention.out.weight',
            f'{src_pfx}.attn.proj.weight',
            f'{src_pfx}.self_attn.out_proj.weight',
            f'{src_pfx}.attention.output.weight',
        )
        if out_src and tgt_out_key in target_sd:
            mapped[tgt_out_key] = source_sd[out_src]
            used_source_keys.add(out_src)
            out_b = out_src.replace('.weight', '.bias')
            if out_b in all_keys:
                used_source_keys.add(out_b)
                skipped_biases.append(out_b)

        # -- Norm 2 --
        for sfx in ['weight', 'bias']:
            src_k = find_key_exact(
                f'{src_pfx}.norm2.{sfx}',
                f'{src_pfx}.layernorm_after.{sfx}',
                f'{src_pfx}.output.LayerNorm.{sfx}',
            )
            tgt_k = f'{tgt_pfx}.norm2.{sfx}'
            if src_k and tgt_k in target_sd:
                mapped[tgt_k] = source_sd[src_k]
                used_source_keys.add(src_k)

        # -- MLP fc1 -> mlp.0 --
        for sfx in ['weight', 'bias']:
            src_k = find_key_exact(
                f'{src_pfx}.mlp.fc1.{sfx}',
                f'{src_pfx}.intermediate.dense.{sfx}',
                f'{src_pfx}.mlp.dense_h_to_4h.{sfx}',
                f'{src_pfx}.mlp.w1.{sfx}',
            )
            tgt_k = f'{tgt_pfx}.mlp.0.{sfx}'
            if src_k and tgt_k in target_sd:
                src_w = source_sd[src_k]
                tgt_w = target_sd[tgt_k]
                if src_w.shape == tgt_w.shape:
                    mapped[tgt_k] = src_w
                else:
                    print(f"    MLP fc1 shape mismatch L{i}: {src_w.shape} vs {tgt_w.shape}")
                    mapped[tgt_k] = tgt_w.clone()
                    slices = tuple(slice(0, min(s, t)) for s, t in zip(src_w.shape, tgt_w.shape))
                    mapped[tgt_k][slices] = src_w[slices]
                used_source_keys.add(src_k)

        # -- MLP fc2 -> mlp.3 --
        for sfx in ['weight', 'bias']:
            src_k = find_key_exact(
                f'{src_pfx}.mlp.fc2.{sfx}',
                f'{src_pfx}.output.dense.{sfx}',
                f'{src_pfx}.mlp.dense_4h_to_h.{sfx}',
                f'{src_pfx}.mlp.w2.{sfx}',
            )
            tgt_k = f'{tgt_pfx}.mlp.3.{sfx}'
            if src_k and tgt_k in target_sd:
                src_w = source_sd[src_k]
                tgt_w = target_sd[tgt_k]
                if src_w.shape == tgt_w.shape:
                    mapped[tgt_k] = src_w
                else:
                    print(f"    MLP fc2 shape mismatch L{i}: {src_w.shape} vs {tgt_w.shape}")
                    mapped[tgt_k] = tgt_w.clone()
                    slices = tuple(slice(0, min(s, t)) for s, t in zip(src_w.shape, tgt_w.shape))
                    mapped[tgt_k][slices] = src_w[slices]
                used_source_keys.add(src_k)

    # ==================================================================
    # 3. FINAL LAYER NORM
    # ==================================================================
    for sfx in ['weight', 'bias']:
        src_k = find_key_exact(
            f'norm.{sfx}', f'layernorm.{sfx}', f'encoder.norm.{sfx}',
            f'ln_post.{sfx}', f'fc_norm.{sfx}',
        )
        tgt_k = f'norm.{sfx}'
        if src_k and tgt_k in target_sd:
            mapped[tgt_k] = source_sd[src_k]
            used_source_keys.add(src_k)

    # ==================================================================
    # 4. SKIP known non-mappable keys
    # ==================================================================
    for k in all_keys:
        if any(pat in k for pat in [
            'cls_token', 'position_embed', 'pos_embed',
            'mask_token', 'decoder', 'head', 'predictor',
        ]):
            used_source_keys.add(k)

    # ==================================================================
    # REPORT
    # ==================================================================
    unmapped = all_keys - used_source_keys
    target_keys = set(target_sd.keys())
    loaded_keys = set(mapped.keys())
    rope_keys = {k for k in target_keys if 'rope.' in k}
    missing_real = target_keys - loaded_keys - rope_keys

    print(f"\n    -- Mapping Summary --")
    print(f"    Source params:       {len(all_keys)}")
    print(f"    Target params:       {len(target_keys)} ({len(rope_keys)} RoPE buffers)")
    print(f"    Successfully mapped: {len(mapped)}")
    print(f"    Skipped biases:      {len(skipped_biases)} (target has bias=False)")
    print(f"    Unmapped source:     {len(unmapped)}")
    if unmapped:
        for k in sorted(unmapped)[:5]:
            print(f"      {k}")
        if len(unmapped) > 5:
            print(f"      ... and {len(unmapped)-5} more")
    print(f"    Missing target:      {len(missing_real)} (randomly initialized)")
    if missing_real:
        for k in sorted(missing_real)[:5]:
            print(f"      {k}")

    coverage = len(mapped) / max(len(target_keys) - len(rope_keys), 1) * 100
    print(f"    Coverage:            {coverage:.1f}%")
    if coverage < 50:
        print(f"    WARNING: LOW COVERAGE! Use --inspect_vjepa2_keys to debug")

    return mapped


# ============================================================================
# V-JEPA 2 BACKBONE DOWNLOAD + MAPPING
# ============================================================================

def download_and_map_vjepa2(model_size, resolution, target_encoder, inspect_keys=False):
    config_map = {
        'vitl': {'embed_dim': 1024, 'depth': 24, 'num_heads': 16, 'hub': 'vjepa2_vitl'},
        'vith': {'embed_dim': 1280, 'depth': 32, 'num_heads': 16, 'hub': 'vjepa2_vith'},
        'vitg': {'embed_dim': 1024, 'depth': 40, 'num_heads': 16, 'hub': 'vjepa2_vitg'},
    }
    cfg = config_map[model_size]
    print(f"\n  Downloading V-JEPA 2 {model_size.upper()} "
          f"({cfg['embed_dim']}d, {cfg['depth']}L, {cfg['num_heads']}H)")

    raw_sd = None

    # Try torch.hub
    try:
        print(f"  -> torch.hub: facebookresearch/vjepa2 / {cfg['hub']}")
        encoder = torch.hub.load('facebookresearch/vjepa2', cfg['hub'], trust_repo=True)
        raw_sd = encoder.state_dict()
        print(f"  Loaded from torch.hub ({len(raw_sd)} tensors)")
    except Exception as e:
        print(f"  torch.hub failed: {e}")

    # Fallback: HuggingFace
    if raw_sd is None:
        try:
            from transformers import AutoModel
            hf_id = f'facebook/vjepa2-{model_size}-fpc64-{resolution}'
            print(f"  -> HuggingFace: {hf_id}")
            model = AutoModel.from_pretrained(hf_id, trust_remote_code=True)
            if hasattr(model, 'encoder'):
                raw_sd = model.encoder.state_dict()
            elif hasattr(model, 'visual'):
                raw_sd = model.visual.state_dict()
            else:
                raw_sd = model.state_dict()
            print(f"  Loaded from HuggingFace ({len(raw_sd)} tensors)")
            del model
        except Exception as e2:
            raise RuntimeError(
                f"Could not download V-JEPA 2.\n"
                f"  Install: pip install transformers\n"
                f"  Or download from: https://github.com/facebookresearch/vjepa2\n"
                f"  Error: {e2}"
            )

    # Strip common prefixes
    cleaned_sd = {}
    for k, v in raw_sd.items():
        new_k = k
        for prefix in ['encoder.', 'module.', 'visual.', 'backbone.']:
            if new_k.startswith(prefix):
                new_k = new_k[len(prefix):]
        cleaned_sd[new_k] = v

    if inspect_keys:
        print(f"\n  -- Source state dict keys ({len(cleaned_sd)}) --")
        for k in sorted(cleaned_sd.keys()):
            print(f"    {k:60s}  {tuple(cleaned_sd[k].shape)}")
        print()

    print(f"\n  Mapping V-JEPA 2 -> VideoViTEncoder ...")
    mapped_sd = map_vjepa2_state_dict(cleaned_sd, target_encoder, cfg['embed_dim'])

    return mapped_sd, cfg


# ============================================================================
# DROID DATASET LOADERS (with >=4s filter)
# ============================================================================

class ManipulationVideoAugmentation:
    def __init__(self, img_size=224, training=True, strength='medium'):
        self.img_size = img_size
        self.training = training
        if training:
            cfgs = {
                'light':  {'scale': (0.9, 1.0), 'rot': 5,  'bri': 0.1, 'con': 0.1},
                'medium': {'scale': (0.7, 1.0), 'rot': 10, 'bri': 0.2, 'con': 0.15},
                'heavy':  {'scale': (0.6, 1.0), 'rot': 15, 'bri': 0.3, 'con': 0.2},
            }
            c = cfgs.get(strength, cfgs['medium'])
            self.spatial_transform = transforms.Compose([
                transforms.RandomResizedCrop(img_size, scale=c['scale']),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=c['rot']),
                transforms.ColorJitter(brightness=c['bri'], contrast=c['con']),
            ])
        else:
            self.spatial_transform = transforms.Compose([
                transforms.Resize(img_size), transforms.CenterCrop(img_size),
            ])
        self.normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    def __call__(self, video_frames):
        first = transforms.ToPILImage()(video_frames[0]) if isinstance(video_frames[0], np.ndarray) else video_frames[0]
        if self.training:
            crop_params = transforms.RandomResizedCrop.get_params(
                first, self.spatial_transform.transforms[0].scale,
                self.spatial_transform.transforms[0].ratio)
        augmented = []
        for frame in video_frames:
            if isinstance(frame, np.ndarray):
                frame = transforms.ToPILImage()(frame)
            if self.training:
                frame = transforms.functional.resized_crop(
                    frame, *crop_params, (self.img_size, self.img_size))
                for t in self.spatial_transform.transforms[1:]:
                    frame = t(frame)
            else:
                frame = self.spatial_transform(frame)
            frame = self.normalize(transforms.ToTensor()(frame))
            augmented.append(frame)
        return torch.stack(augmented, dim=0).permute(1, 0, 2, 3)


class DROIDDatasetRLDS(Dataset):
    """
    DROID RLDS loader. Requires GCS auth OR local copy.

    To download locally:
        pip install gsutil
        gsutil -m cp -r gs://gresearch/robotics/droid ./data/droid_rlds/

    Or switch to HuggingFace format (no auth needed):
        --format huggingface --data_path cadene/droid
    """
    def __init__(self, data_path, num_frames=16, img_size=224,
                 camera='exterior_image_1_left', training=True,
                 augmentation_strength='medium', max_episodes=None,
                 predict_horizon=4):
        self.num_frames = num_frames
        self.img_size = img_size
        self.camera = camera
        self.training = training
        self.predict_horizon = predict_horizon
        self.clip_length = num_frames + predict_horizon

        try:
            import tensorflow_datasets as tfds
        except ImportError:
            raise ImportError(
                "RLDS requires: pip install tensorflow tensorflow_datasets\n"
                "Or use: --format huggingface --data_path cadene/droid")

        is_gcs = str(data_path).startswith('gs://')
        if is_gcs:
            print(f"  [RLDS] GCS path: {data_path}")
            print(f"  NOTE: Requires 'gcloud auth application-default login'")
            print(f"  Or download locally / use --format huggingface")

        # tfds.load("droid", data_dir=X) expects X/droid/version/
        data_path = str(data_path)
        if data_path.rstrip('/').endswith('/droid'):
            data_dir = str(Path(data_path.rstrip('/')).parent)
            ds_name = 'droid'
        elif data_path.rstrip('/').endswith('/droid_100'):
            data_dir = str(Path(data_path.rstrip('/')).parent)
            ds_name = 'droid_100'
        else:
            data_dir = data_path
            ds_name = 'droid'

        try:
            ds = tfds.load(ds_name, data_dir=data_dir, split="train")
        except Exception as e:
            try:
                ds = tfds.load("droid_100", data_dir=data_dir, split="train")
                print(f"  Loaded droid_100 subset")
            except:
                raise RuntimeError(
                    f"Failed to load RLDS from {data_path}: {e}\n\n"
                    f"Alternatives:\n"
                    f"  --format huggingface --data_path cadene/droid\n"
                    f"  --format hdf5 --data_path ./data/droid_hdf5/")

        self.episodes = []
        total_frames = 0
        skipped = 0
        for episode in ds:
            if max_episodes and len(self.episodes) >= max_episodes:
                break
            steps = list(episode['steps'])
            ep_len = len(steps)
            if ep_len / DROID_FPS < MIN_EPISODE_SECONDS or ep_len < self.clip_length:
                skipped += 1
                continue
            self.episodes.append(steps)
            total_frames += ep_len
            if total_frames >= TARGET_FRAMES:
                break

        self.samples = self._create_samples()
        self.augmentation = ManipulationVideoAugmentation(img_size, training, augmentation_strength)
        print(f"  Eps: {len(self.episodes)} (skipped {skipped}), "
              f"{total_frames/DROID_FPS/3600:.1f}h, {len(self.samples)} clips")

    def _create_samples(self):
        samples = []
        stride = max(1, self.num_frames // 2)
        for ep_idx, steps in enumerate(self.episodes):
            for start in range(0, len(steps) - self.clip_length + 1, stride):
                samples.append({'episode_idx': ep_idx, 'start': start})
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        steps = self.episodes[s['episode_idx']]
        frames, actions, states = [], [], []
        for i in range(self.clip_length):
            step = steps[s['start'] + i]
            frames.append(step['observation'][self.camera].numpy())
            actions.append(step['action'].numpy())
            joint = step['observation']['joint_position'].numpy()
            cart = step['observation']['cartesian_position'].numpy()
            grip = step['observation']['gripper_position'].numpy()
            states.append(np.concatenate([joint, cart, grip]))
        video = self.augmentation(frames[:self.num_frames])
        fut_video = self.augmentation(frames[self.num_frames:]) if self.predict_horizon > 0 else None
        actions = torch.from_numpy(np.stack(actions)).float()
        states = torch.from_numpy(np.stack(states)).float()
        result = {
            'video': video, 'actions': actions[:self.num_frames],
            'states': states[:self.num_frames],
            'future_actions': actions[self.num_frames:],
            'future_states': states[self.num_frames:],
            'task_label': torch.tensor(1, dtype=torch.long),
        }
        if fut_video is not None:
            result['future_video'] = fut_video
        return result


class DROIDDatasetHuggingFace(Dataset):
    """DROID HuggingFace LeRobot loader with >=4s filter. No auth needed."""
    def __init__(self, data_path='cadene/droid', num_frames=16, img_size=224,
                 camera='observation.images.exterior_image_1_left',
                 training=True, augmentation_strength='medium',
                 max_episodes=None, predict_horizon=4):
        self.num_frames = num_frames
        self.img_size = img_size
        self.camera = camera
        self.training = training
        self.predict_horizon = predict_horizon
        self.clip_length = num_frames + predict_horizon
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("pip install datasets")

        print(f"  [HuggingFace] {data_path} (streaming, no full download)")
        ds = load_dataset(data_path, split="train", streaming=True)
        self.episodes = []
        current_ep, current_ep_idx = [], -1
        total_frames, skipped, count = 0, 0, 0
        for row in ds:
            ep_idx = row.get('episode_index', 0)
            if ep_idx != current_ep_idx:
                if current_ep:
                    if len(current_ep)/DROID_FPS >= MIN_EPISODE_SECONDS and len(current_ep) >= self.clip_length:
                        self.episodes.append(current_ep)
                        total_frames += len(current_ep)
                    else:
                        skipped += 1
                    if max_episodes and len(self.episodes) >= max_episodes:
                        break
                    if total_frames >= TARGET_FRAMES:
                        break
                current_ep, current_ep_idx = [], ep_idx
            current_ep.append(row)
            count += 1
            if count % 50000 == 0:
                print(f"    {count} frames, {len(self.episodes)} eps, "
                      f"{total_frames/DROID_FPS/3600:.1f}h")
        if current_ep and len(current_ep)/DROID_FPS >= MIN_EPISODE_SECONDS:
            self.episodes.append(current_ep)
            total_frames += len(current_ep)

        self.samples = self._create_samples()
        self.augmentation = ManipulationVideoAugmentation(img_size, training, augmentation_strength)
        print(f"  Eps: {len(self.episodes)} (skipped {skipped}), "
              f"{total_frames/DROID_FPS/3600:.1f}h, {len(self.samples)} clips")

    def _create_samples(self):
        samples = []
        stride = max(1, self.num_frames // 2)
        for ep_idx, ep in enumerate(self.episodes):
            for start in range(0, len(ep) - self.clip_length + 1, stride):
                samples.append({'episode_idx': ep_idx, 'start': start})
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        ep = self.episodes[s['episode_idx']]
        frames, actions, states = [], [], []
        for i in range(self.clip_length):
            row = ep[s['start'] + i]
            img = np.array(row[self.camera])
            if img.ndim == 2:
                img = np.stack([img]*3, axis=-1)
            frames.append(img)
            actions.append(np.array(row['action'], dtype=np.float32))
            states.append(np.array(row['observation.state'], dtype=np.float32))
        video = self.augmentation(frames[:self.num_frames])
        fut_video = self.augmentation(frames[self.num_frames:]) if self.predict_horizon > 0 else None
        actions = torch.from_numpy(np.stack(actions)).float()
        states = torch.from_numpy(np.stack(states)).float()
        result = {
            'video': video, 'actions': actions[:self.num_frames],
            'states': states[:self.num_frames],
            'future_actions': actions[self.num_frames:],
            'future_states': states[self.num_frames:],
            'task_label': torch.tensor(1, dtype=torch.long),
        }
        if fut_video is not None:
            result['future_video'] = fut_video
        return result


class DROIDDatasetHDF5(Dataset):
    """DROID HDF5 loader with >=4s filter."""
    def __init__(self, data_path, num_frames=16, img_size=224,
                 camera='exterior_image_1_left', training=True,
                 augmentation_strength='medium', max_episodes=None,
                 predict_horizon=4):
        self.data_path = Path(data_path)
        self.num_frames = num_frames
        self.img_size = img_size
        self.camera = camera
        self.training = training
        self.predict_horizon = predict_horizon
        self.clip_length = num_frames + predict_horizon
        try:
            import h5py; self.h5py = h5py
        except ImportError:
            raise ImportError("pip install h5py")

        all_files = sorted(self.data_path.glob('**/*.hdf5')) + sorted(self.data_path.glob('**/*.h5'))
        if not all_files:
            raise FileNotFoundError(f"No HDF5 files in {data_path}")

        self.hdf5_files = []
        total_frames, skipped = 0, 0
        for fpath in all_files:
            ep_len = self._get_episode_length(fpath)
            if ep_len / DROID_FPS < MIN_EPISODE_SECONDS or ep_len < self.clip_length:
                skipped += 1; continue
            self.hdf5_files.append(fpath)
            total_frames += ep_len
            if max_episodes and len(self.hdf5_files) >= max_episodes:
                break
            if total_frames >= TARGET_FRAMES:
                break

        self.samples = self._create_samples()
        self.augmentation = ManipulationVideoAugmentation(img_size, training, augmentation_strength)
        print(f"  [HDF5] Eps: {len(self.hdf5_files)} (skipped {skipped}), "
              f"{total_frames/DROID_FPS/3600:.1f}h, {len(self.samples)} clips")

    def _get_episode_length(self, fpath):
        try:
            with self.h5py.File(fpath, 'r') as f:
                for key in ['action', 'actions', 'action/cartesian_position']:
                    if key in f: return f[key].shape[0]
            return 0
        except:
            return 0

    def _create_samples(self):
        samples = []
        stride = max(1, self.num_frames // 2)
        for fi, fpath in enumerate(self.hdf5_files):
            ep_len = self._get_episode_length(fpath)
            for start in range(0, ep_len - self.clip_length + 1, stride):
                samples.append({'file_idx': fi, 'start': start})
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        fpath = self.hdf5_files[s['file_idx']]
        start = s['start']
        end = start + self.clip_length
        with self.h5py.File(fpath, 'r') as f:
            img_data = None
            for img_key in [f'obs/{self.camera}',
                            f'observation/{self.camera}/image_left',
                            f'observation/exterior_image_1/image_left',
                            self.camera]:
                if img_key in f:
                    img_data = f[img_key][start:end]; break
            if img_data is None:
                def find_img(g, pfx=''):
                    for k in g.keys():
                        p = f"{pfx}/{k}" if pfx else k
                        if isinstance(g[k], self.h5py.Dataset) and g[k].ndim == 4: return p
                        elif isinstance(g[k], self.h5py.Group):
                            r = find_img(g[k], p)
                            if r: return r
                    return None
                ik = find_img(f)
                img_data = f[ik][start:end] if ik else np.zeros((self.clip_length, 224, 224, 3), dtype=np.uint8)
            frames = [img_data[i] for i in range(self.clip_length)]

            if 'action' in f and f['action'].ndim == 2:
                raw_act = f['action'][start:end]
            elif 'action/cartesian_position' in f:
                raw_act = np.concatenate([f['action/cartesian_position'][start:end],
                                          f['action/gripper_position'][start:end]], axis=-1)
            else:
                raw_act = np.zeros((self.clip_length, 7), dtype=np.float32)

            if 'obs/joint_positions' in f:
                raw_st = np.concatenate([f['obs/joint_positions'][start:end],
                                         f['obs/cartesian_position'][start:end],
                                         f['obs/gripper_position'][start:end]], axis=-1)
            elif 'observation/robot_state/joint_positions' in f:
                raw_st = np.concatenate([f['observation/robot_state/joint_positions'][start:end],
                                         f['observation/robot_state/cartesian_position'][start:end],
                                         f['observation/robot_state/gripper_position'][start:end]], axis=-1)
            else:
                raw_st = np.zeros((self.clip_length, 14), dtype=np.float32)

        video = self.augmentation(frames[:self.num_frames])
        fut_video = self.augmentation(frames[self.num_frames:]) if self.predict_horizon > 0 else None
        actions = torch.from_numpy(raw_act.astype(np.float32))
        states = torch.from_numpy(raw_st.astype(np.float32))
        result = {
            'video': video, 'actions': actions[:self.num_frames],
            'states': states[:self.num_frames],
            'future_actions': actions[self.num_frames:],
            'future_states': states[self.num_frames:],
            'task_label': torch.tensor(1, dtype=torch.long),
        }
        if fut_video is not None:
            result['future_video'] = fut_video
        return result


def create_dataset(args):
    common = dict(num_frames=args.num_frames, img_size=args.img_size, training=True,
                  augmentation_strength=args.augmentation_strength,
                  max_episodes=args.max_episodes, predict_horizon=args.predict_horizon)
    if args.format == 'rlds':
        return DROIDDatasetRLDS(args.data_path, camera=args.camera, **common)
    elif args.format == 'huggingface':
        return DROIDDatasetHuggingFace(args.data_path, camera=f'observation.images.{args.camera}', **common)
    elif args.format == 'hdf5':
        return DROIDDatasetHDF5(args.data_path, camera=args.camera, **common)
    raise ValueError(f"Unknown format: {args.format}")


# ============================================================================
# EMA TARGET ENCODER
# ============================================================================

class EMAEncoder:
    def __init__(self, encoder, decay=0.999):
        self.ema_encoder = copy.deepcopy(encoder)
        self.ema_encoder.requires_grad_(False)
        self.decay = decay
    def update(self, encoder):
        with torch.no_grad():
            for ep, p in zip(self.ema_encoder.parameters(), encoder.parameters()):
                ep.data.mul_(self.decay).add_(p.data, alpha=1-self.decay)
    @torch.no_grad()
    def encode(self, video):
        self.ema_encoder.eval()
        return self.ema_encoder(video)
    def to(self, device):
        self.ema_encoder.to(device); return self
    def state_dict(self):
        return self.ema_encoder.state_dict()
    def load_state_dict(self, sd):
        self.ema_encoder.load_state_dict(sd)


# ============================================================================
# LOSSES
# ============================================================================

def latent_prediction_loss(pred, tgt):
    return F.smooth_l1_loss(F.normalize(pred, dim=-1), F.normalize(tgt, dim=-1))
def action_prediction_loss(pred, gt):
    return F.mse_loss(pred, gt)
def state_prediction_loss(pred, gt):
    return F.mse_loss(pred, gt)
def task_classification_loss(logits, labels):
    return F.cross_entropy(logits, labels)
def uncertainty_regularization(unc, min_val=0.01):
    return F.relu(min_val - unc.mean())


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train(model, ema, dataloader, args):
    device = args.device
    model.to(device); ema.to(device)
    save_dir = Path(args.save_dir); save_dir.mkdir(parents=True, exist_ok=True)

    encoder_params = list(model.encoder.parameters())
    predictor_params = [p for n, p in model.named_parameters() if not n.startswith('encoder.')]
    optimizer = torch.optim.AdamW([
        {'params': encoder_params, 'lr': args.encoder_lr},
        {'params': predictor_params, 'lr': args.learning_rate},
    ], weight_decay=args.weight_decay)

    total_steps = len(dataloader) * args.num_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    def lr_fn(step):
        if step < warmup_steps: return step / max(warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / max(total_steps - warmup_steps, 1)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)

    use_amp = args.use_amp and device == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    config = {**vars(args), 'total_steps': total_steps, 'warmup_steps': warmup_steps}
    with open(save_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print(f"TRAINING: {len(dataloader.dataset)} clips, {args.num_epochs} epochs, "
          f"{total_steps} steps")
    print(f"  enc_lr={args.encoder_lr} pred_lr={args.learning_rate} "
          f"warmup={warmup_steps} device={device} amp={use_amp}")
    print(f"{'='*70}\n")

    best_loss, global_step = float('inf'), 0
    for epoch in range(1, args.num_epochs + 1):
        model.train()
        ep_loss = {k: 0.0 for k in ['total','latent','action','state','task','unc']}
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.num_epochs}")
        for batch in pbar:
            video = batch['video'].to(device)
            act = torch.cat([batch['actions'], batch['future_actions']], 1).to(device)
            st = torch.cat([batch['states'], batch['future_states']], 1).to(device)
            fut_act = batch['future_actions'].to(device)
            fut_st = batch['future_states'].to(device)
            task_lbl = batch['task_label'].to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device, dtype=torch.float16 if use_amp else torch.float32, enabled=use_amp):
                preds, cur_log, fut_log = model(video, act, st,
                    predict_horizon=args.predict_horizon, encoder_frozen=False)
                with torch.no_grad():
                    tgt_feat = ema.encode(video).mean(dim=1)

                l_lat = sum(latent_prediction_loss(p['features'], tgt_feat) for p in preds) / max(len(preds),1)
                l_act = sum(action_prediction_loss(p['action'], fut_act[:,h]) for h,p in enumerate(preds) if h < fut_act.shape[1]) / max(len(preds),1)
                l_st = sum(state_prediction_loss(p['state'], fut_st[:,h]) for h,p in enumerate(preds) if h < fut_st.shape[1]) / max(len(preds),1)
                l_task = task_classification_loss(cur_log, task_lbl)
                if fut_log is not None:
                    l_task = (l_task + task_classification_loss(fut_log, task_lbl)) / 2
                l_unc = sum(uncertainty_regularization(p['action_uncertainty']) + uncertainty_regularization(p['state_uncertainty']) for p in preds) / max(2*len(preds),1)
                loss = (args.lambda_latent*l_lat + args.lambda_action*l_act +
                        args.lambda_state*l_st + args.lambda_task*l_task +
                        args.lambda_uncertainty*l_unc)

            if use_amp:
                scaler.scale(loss).backward(); scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer); scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
            scheduler.step(); ema.update(model.encoder); global_step += 1

            ep_loss['total'] += loss.item(); ep_loss['latent'] += l_lat.item()
            ep_loss['action'] += l_act.item(); ep_loss['state'] += l_st.item()
            ep_loss['task'] += l_task.item(); ep_loss['unc'] += l_unc.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", lat=f"{l_lat.item():.3f}",
                             act=f"{l_act.item():.3f}", lr=f"{scheduler.get_last_lr()[1]:.1e}")

            if global_step % args.save_every_steps == 0:
                torch.save({'epoch':epoch,'global_step':global_step,
                    'model_state_dict':model.state_dict(),
                    'ema_encoder_state_dict':ema.state_dict(),
                    'optimizer_state_dict':optimizer.state_dict()},
                    save_dir / f"step_{global_step}.pth")

        n = max(len(dataloader), 1)
        avg = {k: v/n for k,v in ep_loss.items()}
        print(f"\n  Epoch {epoch}: total={avg['total']:.4f} lat={avg['latent']:.4f} "
              f"act={avg['action']:.4f} st={avg['state']:.4f} task={avg['task']:.4f}")

        ckpt = {'epoch':epoch, 'global_step':global_step,
                'model_state_dict':model.state_dict(),
                'ema_encoder_state_dict':ema.state_dict(),
                'optimizer_state_dict':optimizer.state_dict(),
                'scheduler_state_dict':scheduler.state_dict(),
                'epoch_losses':avg, 'config':config}
        torch.save(ckpt, save_dir / f"epoch_{epoch:03d}.pth")
        if avg['total'] < best_loss:
            best_loss = avg['total']
            torch.save(ckpt, save_dir / "best_model.pth")
            print(f"  * Best model (loss={best_loss:.4f})")

    torch.save(ckpt, save_dir / "final_model.pth")
    print(f"\nDONE. Best={best_loss:.4f}  Checkpoints: {save_dir}\n")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train Action-Conditioned World Model on DROID')
    g = parser.add_argument_group('Data')
    g.add_argument('--data_path', type=str, required=True)
    g.add_argument('--format', type=str, default='hdf5', choices=['rlds','huggingface','hdf5'])
    g.add_argument('--camera', type=str, default='exterior_image_1_left')
    g.add_argument('--max_episodes', type=int, default=None)
    g.add_argument('--augmentation_strength', type=str, default='medium', choices=['light','medium','heavy'])

    g = parser.add_argument_group('V-JEPA 2')
    g.add_argument('--vjepa2_size', type=str, default='vitl', choices=['vitl','vith','vitg'])
    g.add_argument('--vjepa2_resolution', type=int, default=256, choices=[256,384])
    g.add_argument('--pretrained_checkpoint', type=str, default=None)
    g.add_argument('--inspect_vjepa2_keys', action='store_true')

    g = parser.add_argument_group('Model')
    g.add_argument('--img_size', type=int, default=224)
    g.add_argument('--num_frames', type=int, default=16)
    g.add_argument('--predictor_dim', type=int, default=1024)
    g.add_argument('--predictor_depth', type=int, default=12)
    g.add_argument('--predictor_heads', type=int, default=8)
    g.add_argument('--action_dim', type=int, default=7)
    g.add_argument('--state_dim', type=int, default=14)
    g.add_argument('--num_task_classes', type=int, default=2)
    g.add_argument('--predict_horizon', type=int, default=4)

    g = parser.add_argument_group('Training')
    g.add_argument('--num_epochs', type=int, default=50)
    g.add_argument('--batch_size', type=int, default=4)
    g.add_argument('--learning_rate', type=float, default=3e-4)
    g.add_argument('--encoder_lr', type=float, default=1e-5)
    g.add_argument('--weight_decay', type=float, default=0.05)
    g.add_argument('--warmup_ratio', type=float, default=0.05)
    g.add_argument('--grad_clip', type=float, default=1.0)
    g.add_argument('--ema_decay', type=float, default=0.999)
    g.add_argument('--num_workers', type=int, default=4)
    g.add_argument('--use_amp', action='store_true', default=False)
    g.add_argument('--use_grad_checkpoint', action='store_true', default=False)

    g = parser.add_argument_group('Loss')
    g.add_argument('--lambda_latent', type=float, default=1.0)
    g.add_argument('--lambda_action', type=float, default=1.0)
    g.add_argument('--lambda_state', type=float, default=0.5)
    g.add_argument('--lambda_task', type=float, default=0.1)
    g.add_argument('--lambda_uncertainty', type=float, default=0.01)

    g = parser.add_argument_group('I/O')
    g.add_argument('--save_dir', type=str, default='./checkpoints/world_model')
    g.add_argument('--save_every_steps', type=int, default=5000)
    g.add_argument('--device', type=str, default='auto')
    g.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else ('mps' if hasattr(torch.backends,'mps') and torch.backends.mps.is_available() else 'cpu')

    print(f"\n{'='*70}")
    print(f"DROID ACTION-CONDITIONED WORLD MODEL")
    print(f"  Backbone: V-JEPA 2 {args.vjepa2_size.upper()}")
    print(f"  Data: {args.data_path} ({args.format})")
    print(f"  Filter: >= {MIN_EPISODE_SECONDS}s | Target: {TARGET_HOURS}h | Device: {args.device}")
    print(f"{'='*70}")

    # 1. Build model
    vcfg = {'vitl':{'embed_dim':1024,'depth':24,'num_heads':16},
            'vith':{'embed_dim':1280,'depth':32,'num_heads':16},
            'vitg':{'embed_dim':1024,'depth':40,'num_heads':16}}[args.vjepa2_size]

    print(f"\n[1/4] Building model ({vcfg['embed_dim']}d x {vcfg['depth']}L encoder)")
    model = DROIDActionConditionedWorldModel(
        img_size=args.img_size, num_frames=args.num_frames, tubelet_size=(2,16,16),
        encoder_dim=vcfg['embed_dim'], encoder_depth=vcfg['depth'], encoder_heads=vcfg['num_heads'],
        predictor_dim=args.predictor_dim, predictor_depth=args.predictor_depth,
        predictor_heads=args.predictor_heads, action_dim=args.action_dim, state_dim=args.state_dim,
        num_task_classes=args.num_task_classes, temporal_scale=1.0,
        use_grad_checkpoint=args.use_grad_checkpoint)
    print(f"  Total: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

    # 2. Load V-JEPA 2 weights
    if args.pretrained_checkpoint:
        print(f"\n[2/4] Loading checkpoint: {args.pretrained_checkpoint}")
        ckpt = torch.load(args.pretrained_checkpoint, map_location='cpu', weights_only=False)
        if 'model_state_dict' in ckpt:
            m, u = model.load_state_dict(ckpt['model_state_dict'], strict=False)
            print(f"  Full model (missing={len(m)}, unexpected={len(u)})")
        else:
            m, u = model.encoder.load_state_dict(ckpt.get('encoder_state_dict',{}), strict=False)
            print(f"  Encoder only (missing={len(m)}, unexpected={len(u)})")
    else:
        print(f"\n[2/4] Downloading V-JEPA 2 + mapping weights ...")
        mapped_sd, _ = download_and_map_vjepa2(
            args.vjepa2_size, args.vjepa2_resolution,
            model.encoder, inspect_keys=args.inspect_vjepa2_keys)
        m, u = model.encoder.load_state_dict(mapped_sd, strict=False)
        real_missing = [k for k in m if 'rope.' not in k]
        n_target = len([k for k in model.encoder.state_dict() if 'rope.' not in k])
        print(f"\n  Loaded {len(mapped_sd)}/{n_target} encoder params")
        if real_missing:
            print(f"  Still missing: {real_missing[:5]}{'...' if len(real_missing)>5 else ''}")

    ema = EMAEncoder(model.encoder, decay=args.ema_decay)
    print(f"  EMA target encoder (decay={args.ema_decay})")

    # 3. Dataset
    print(f"\n[3/4] Loading dataset ...")
    dataset = create_dataset(args)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(args.device=='cuda'),
        drop_last=True, persistent_workers=(args.num_workers > 0))
    print(f"  {len(dataset)} clips, {len(dataloader)} batches/epoch")

    # 4. Train
    print(f"\n[4/4] Training ...")
    train(model, ema, dataloader, args)


if __name__ == "__main__":
    main()

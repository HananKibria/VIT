"""
STAGE 1: DROID Pre-training (Standalone Script)
================================================

Pre-trains the video encoder on the DROID dataset using:
1. Temporal order prediction (shuffled vs. correct sequence)
2. Symmetric InfoNCE contrastive learning
3. Optional: Structured temporal mask denoising

DROID Dataset:
- 76k manipulation trajectories, 350 hours, 15 Hz
- Franka Panda 7-DOF with Robotiq gripper
- 3 cameras: 2 exterior Zed 2 (180x320) + 1 wrist Zed Mini (180x320)
- Each episode: RGB video + actions + states + language instructions
- Formats: RLDS (primary), HuggingFace LeRobot, or local HDF5

Usage:
    # From RLDS (Google Cloud):
    python stage1_droid_pretraining.py --data_path gs://gresearch/robotics/droid --format rlds

    # From HuggingFace:
    python stage1_droid_pretraining.py --data_path cadene/droid --format huggingface

    # From local directory of HDF5 files:
    python stage1_droid_pretraining.py --data_path ./data/droid_hdf5 --format hdf5

    # Small debug set (100 episodes):
    python stage1_droid_pretraining.py --data_path gs://gresearch/robotics/droid_100 --format rlds
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm
import argparse
import os
import json

from droid_world_model import DROIDActionConditionedWorldModel


# ============================================================================
# DROID DATASET IMPLEMENTATIONS
# ============================================================================

class DROIDDatasetRLDS(Dataset):
    """
    DROID dataset loader using RLDS format (TensorFlow Datasets).

    Loads episodes from the RLDS-formatted DROID dataset at
    gs://gresearch/robotics/droid or a local copy.

    Each episode step contains:
      observation:
        exterior_image_1_left: (180, 320, 3) uint8
        exterior_image_2_left: (180, 320, 3) uint8
        wrist_image_left:      (180, 320, 3) uint8
        cartesian_position:    (6,)  float64  — EE pose
        joint_position:        (7,)  float64  — joint angles
        gripper_position:      (1,)  float64  — gripper opening
      action:                  (7,)  float64  — target EE + gripper
      language_instruction:    string
    """

    def __init__(self, data_path, num_frames=16, img_size=224,
                 camera='exterior_image_1_left', training=True,
                 augmentation_strength='medium', max_episodes=None):
        self.num_frames = num_frames
        self.img_size = img_size
        self.camera = camera
        self.training = training

        try:
            import tensorflow_datasets as tfds
        except ImportError:
            raise ImportError(
                "tensorflow_datasets required for RLDS format.\n"
                "Install: pip install tensorflow tensorflow_datasets"
            )

        print(f"Loading DROID RLDS from: {data_path}")
        print(f"Camera: {camera}")

        ds = tfds.load("droid", data_dir=data_path, split="train")

        # Pre-index episodes for random access
        self.episodes = []
        for i, episode in enumerate(ds):
            if max_episodes and i >= max_episodes:
                break
            steps = list(episode['steps'])
            if len(steps) >= num_frames:
                self.episodes.append(steps)

        self.samples = self._create_samples()
        self.augmentation = ManipulationVideoAugmentation(img_size, training, augmentation_strength)
        print(f"DROID RLDS Dataset: {len(self.samples)} clips from {len(self.episodes)} episodes")

    def _create_samples(self):
        samples = []
        stride = max(1, self.num_frames // 2)  # 50% overlap
        for ep_idx, steps in enumerate(self.episodes):
            for start in range(0, len(steps) - self.num_frames + 1, stride):
                samples.append({'episode_idx': ep_idx, 'start': start})
        return samples

    def _load_clip(self, episode_idx, start):
        steps = self.episodes[episode_idx]
        frames, actions, states = [], [], []
        for i in range(self.num_frames):
            step = steps[start + i]
            img = step['observation'][self.camera].numpy()
            frames.append(img)
            action = step['action'].numpy()  # (7,)
            cart = step['observation']['cartesian_position'].numpy()  # (6,)
            joint = step['observation']['joint_position'].numpy()     # (7,)
            grip = step['observation']['gripper_position'].numpy()    # (1,)
            state = np.concatenate([joint, cart, grip])               # (14,)
            actions.append(action)
            states.append(state)
        return frames, np.stack(actions), np.stack(states)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        frames, actions, states = self._load_clip(s['episode_idx'], s['start'])
        video = self.augmentation(frames)
        actions = torch.from_numpy(actions).float()
        states = torch.from_numpy(states).float()
        if self.training:
            shuffled_video, order_label = self._temporal_order_task(video)
            video_aug = self.augmentation(frames)
            return {'video': video, 'video_augmented': video_aug,
                    'shuffled_video': shuffled_video, 'order_label': order_label,
                    'actions': actions, 'states': states}
        return {'video': video, 'actions': actions, 'states': states}

    def _temporal_order_task(self, video):
        if np.random.random() < 0.5:
            return video, 0
        C, T, H, W = video.shape
        return video[:, torch.randperm(T), :, :], 1


class DROIDDatasetHuggingFace(Dataset):
    """
    DROID dataset loader using HuggingFace LeRobot format.
    Data from: cadene/droid (400 GB, LeRobot parquet + mp4)

    Features:
      observation.images.exterior_image_1_left  — video (180x320)
      observation.images.exterior_image_2_left  — video (180x320)
      observation.images.wrist_image_left       — video (180x320)
      observation.state   — (14,) = joint(7) + cartesian(6) + gripper(1)
      action              — (7,)  = cartesian target(6) + gripper(1)
    """

    def __init__(self, data_path='cadene/droid', num_frames=16, img_size=224,
                 camera='observation.images.exterior_image_1_left',
                 training=True, augmentation_strength='medium', max_episodes=None):
        self.num_frames = num_frames
        self.img_size = img_size
        self.camera = camera
        self.training = training

        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Install: pip install datasets")

        print(f"Loading DROID from HuggingFace: {data_path}")
        print(f"Camera: {camera}")

        # Load dataset metadata
        self.ds = load_dataset(data_path, split="train", streaming=True)

        # Group frames by episode
        self.episodes = []
        current_ep = []
        current_ep_idx = -1

        count = 0
        for row in self.ds:
            ep_idx = row.get('episode_index', 0)
            if ep_idx != current_ep_idx:
                if len(current_ep) >= num_frames:
                    self.episodes.append(current_ep)
                    if max_episodes and len(self.episodes) >= max_episodes:
                        break
                current_ep = []
                current_ep_idx = ep_idx
            current_ep.append(row)
            count += 1
            if count % 10000 == 0:
                print(f"  Loaded {count} frames, {len(self.episodes)} episodes...")

        if len(current_ep) >= num_frames and (not max_episodes or len(self.episodes) < max_episodes):
            self.episodes.append(current_ep)

        self.samples = self._create_samples()
        self.augmentation = ManipulationVideoAugmentation(img_size, training, augmentation_strength)
        print(f"DROID HF Dataset: {len(self.samples)} clips from {len(self.episodes)} episodes")

    def _create_samples(self):
        samples = []
        stride = max(1, self.num_frames // 2)
        for ep_idx, ep in enumerate(self.episodes):
            for start in range(0, len(ep) - self.num_frames + 1, stride):
                samples.append({'episode_idx': ep_idx, 'start': start})
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        ep = self.episodes[s['episode_idx']]
        frames, actions, states = [], [], []
        for i in range(self.num_frames):
            row = ep[s['start'] + i]
            # Image from video frame
            img = np.array(row[self.camera])
            if img.ndim == 2:
                img = np.stack([img]*3, axis=-1)
            frames.append(img)
            actions.append(np.array(row['action'], dtype=np.float32))
            states.append(np.array(row['observation.state'], dtype=np.float32))

        video = self.augmentation(frames)
        actions = torch.from_numpy(np.stack(actions)).float()
        states = torch.from_numpy(np.stack(states)).float()

        if self.training:
            shuffled_video, order_label = self._temporal_order_task(video)
            video_aug = self.augmentation(frames)
            return {'video': video, 'video_augmented': video_aug,
                    'shuffled_video': shuffled_video, 'order_label': order_label,
                    'actions': actions, 'states': states}
        return {'video': video, 'actions': actions, 'states': states}

    def _temporal_order_task(self, video):
        if np.random.random() < 0.5:
            return video, 0
        return video[:, torch.randperm(video.shape[1])], 1


class DROIDDatasetHDF5(Dataset):
    """
    DROID dataset loader from local HDF5 files.

    Expected structure per HDF5 file (one per episode):
      /action                  (T, 7)   — cartesian target + gripper
      /obs/joint_positions     (T, 7)   — joint angles
      /obs/cartesian_position  (T, 6)   — EE pose
      /obs/gripper_position    (T, 1)   — gripper state
      /obs/<camera_key>        (T, H, W, 3) — RGB images

    OR the raw DROID HDF5 format with:
      /action/cartesian_position  (T, 6)
      /action/gripper_position    (T, 1)
      /observation/<camera>/image_left  (T, H, W, 3)
      /observation/robot_state/joint_positions  (T, 7)
      /observation/robot_state/cartesian_position  (T, 6)
      /observation/robot_state/gripper_position  (T, 1)
    """

    def __init__(self, data_path, num_frames=16, img_size=224,
                 camera='exterior_image_1_left', training=True,
                 augmentation_strength='medium', max_episodes=None):
        self.data_path = Path(data_path)
        self.num_frames = num_frames
        self.img_size = img_size
        self.camera = camera
        self.training = training

        try:
            import h5py
            self.h5py = h5py
        except ImportError:
            raise ImportError("Install: pip install h5py")

        # Discover HDF5 files
        self.hdf5_files = sorted(self.data_path.glob('**/*.hdf5')) + \
                          sorted(self.data_path.glob('**/*.h5'))
        if max_episodes:
            self.hdf5_files = self.hdf5_files[:max_episodes]

        if not self.hdf5_files:
            raise FileNotFoundError(f"No HDF5 files found in {data_path}")

        # Index valid episodes
        self.samples = self._create_samples()
        self.augmentation = ManipulationVideoAugmentation(img_size, training, augmentation_strength)
        print(f"DROID HDF5 Dataset: {len(self.samples)} clips from {len(self.hdf5_files)} episodes")

    def _get_episode_length(self, fpath):
        try:
            with self.h5py.File(fpath, 'r') as f:
                # Try common structures
                for key in ['action', 'actions', 'action/cartesian_position']:
                    if key in f:
                        return f[key].shape[0]
            return 0
        except:
            return 0

    def _create_samples(self):
        samples = []
        stride = max(1, self.num_frames // 2)
        for fi, fpath in enumerate(self.hdf5_files):
            ep_len = self._get_episode_length(fpath)
            for start in range(0, ep_len - self.num_frames + 1, stride):
                samples.append({'file_idx': fi, 'start': start})
        return samples

    def _load_clip_from_hdf5(self, file_idx, start):
        fpath = self.hdf5_files[file_idx]
        frames, actions_list, states_list = [], [], []

        with self.h5py.File(fpath, 'r') as f:
            end = start + self.num_frames

            # --- Detect image key ---
            img_data = None
            for img_key in [
                f'obs/{self.camera}',
                f'observation/{self.camera}/image_left',
                f'observation/exterior_image_1/image_left',
                f'observation/wrist_image/image_left',
                self.camera,
            ]:
                if img_key in f:
                    img_data = f[img_key][start:end]
                    break

            if img_data is None:
                # Fallback: try to find any image-like dataset
                def find_images(group, prefix=''):
                    for k in group.keys():
                        path = f"{prefix}/{k}" if prefix else k
                        if isinstance(group[k], self.h5py.Dataset) and group[k].ndim == 4:
                            return path
                        elif isinstance(group[k], self.h5py.Group):
                            r = find_images(group[k], path)
                            if r: return r
                    return None
                img_key = find_images(f)
                if img_key:
                    img_data = f[img_key][start:end]
                else:
                    raise KeyError(f"No image data found in {fpath}")

            for i in range(self.num_frames):
                frames.append(img_data[i])

            # --- Actions (7D) ---
            if 'action' in f and f['action'].ndim == 2:
                raw_actions = f['action'][start:end]
            elif 'action/cartesian_position' in f:
                cart_a = f['action/cartesian_position'][start:end]  # (T,6)
                grip_a = f['action/gripper_position'][start:end]    # (T,1)
                raw_actions = np.concatenate([cart_a, grip_a], axis=-1)
            elif 'actions' in f:
                raw_actions = f['actions'][start:end]
            else:
                raw_actions = np.zeros((self.num_frames, 7), dtype=np.float32)

            # --- States (14D) ---
            if 'obs/joint_positions' in f:
                joint = f['obs/joint_positions'][start:end]
                cart = f['obs/cartesian_position'][start:end]
                grip = f['obs/gripper_position'][start:end]
                raw_states = np.concatenate([joint, cart, grip], axis=-1)
            elif 'observation/robot_state/joint_positions' in f:
                joint = f['observation/robot_state/joint_positions'][start:end]
                cart = f['observation/robot_state/cartesian_position'][start:end]
                grip = f['observation/robot_state/gripper_position'][start:end]
                raw_states = np.concatenate([joint, cart, grip], axis=-1)
            else:
                raw_states = np.zeros((self.num_frames, 14), dtype=np.float32)

        return frames, raw_actions.astype(np.float32), raw_states.astype(np.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        frames, actions, states = self._load_clip_from_hdf5(s['file_idx'], s['start'])
        video = self.augmentation(frames)
        actions = torch.from_numpy(actions).float()
        states = torch.from_numpy(states).float()

        if self.training:
            shuffled_video, order_label = self._temporal_order_task(video)
            video_aug = self.augmentation(frames)
            return {'video': video, 'video_augmented': video_aug,
                    'shuffled_video': shuffled_video, 'order_label': order_label,
                    'actions': actions, 'states': states}
        return {'video': video, 'actions': actions, 'states': states}

    def _temporal_order_task(self, video):
        if np.random.random() < 0.5:
            return video, 0
        return video[:, torch.randperm(video.shape[1])], 1


# ============================================================================
# VIDEO AUGMENTATION FOR MANIPULATION DATA
# ============================================================================

class ManipulationVideoAugmentation:
    """
    Data augmentation for robot manipulation videos.

    Key differences from surgical augmentation:
    - Horizontal flip IS allowed (manipulation is not laterality-specific)
    - Vertical flip is NOT used (gravity matters for manipulation)
    - Scale range accommodates non-square DROID images (180x320 -> 224x224)
    - Color jitter is moderate (diverse real-world scenes)
    """

    def __init__(self, img_size=224, training=True, strength='medium'):
        self.img_size = img_size
        self.training = training

        if training:
            cfgs = {
                'light':  {'scale': (0.9, 1.0), 'rot': 5,  'bright': 0.1, 'contrast': 0.1},
                'medium': {'scale': (0.7, 1.0), 'rot': 10, 'bright': 0.2, 'contrast': 0.15},
                'heavy':  {'scale': (0.6, 1.0), 'rot': 15, 'bright': 0.3, 'contrast': 0.2},
            }
            c = cfgs.get(strength, cfgs['medium'])
            self.spatial_transform = transforms.Compose([
                transforms.RandomResizedCrop(img_size, scale=c['scale']),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=c['rot']),
                transforms.ColorJitter(brightness=c['bright'], contrast=c['contrast']),
            ])
        else:
            self.spatial_transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.CenterCrop(img_size),
            ])

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __call__(self, video_frames):
        """
        Args:
            video_frames: List of numpy arrays (H, W, 3) uint8
        Returns:
            tensor: (C, T, H, W) float32 normalized
        """
        augmented = []

        # Get consistent spatial parameters for all frames
        if self.training:
            first = transforms.ToPILImage()(video_frames[0]) if isinstance(video_frames[0], np.ndarray) else video_frames[0]
            crop_params = transforms.RandomResizedCrop.get_params(
                first,
                self.spatial_transform.transforms[0].scale,
                self.spatial_transform.transforms[0].ratio,
            )

        for frame in video_frames:
            if isinstance(frame, np.ndarray):
                frame = transforms.ToPILImage()(frame)

            if self.training:
                frame = transforms.functional.resized_crop(
                    frame, *crop_params, (self.img_size, self.img_size))
                # Apply remaining transforms (flip, rotation, jitter)
                for t in self.spatial_transform.transforms[1:]:
                    frame = t(frame)
            else:
                frame = self.spatial_transform(frame)

            frame = transforms.ToTensor()(frame)
            frame = self.normalize(frame)
            augmented.append(frame)

        # (T, C, H, W) -> (C, T, H, W)
        return torch.stack(augmented, dim=0).permute(1, 0, 2, 3)


# ============================================================================
# STRUCTURED TEMPORAL MASKING
# ============================================================================

class StructuredTemporalMasking:
    @staticmethod
    def future_masking(encoded_features, num_temporal_patches, mask_ratio=0.5):
        B, N, D = encoded_features.shape
        spatial_per_frame = N // num_temporal_patches
        cutoff = int(num_temporal_patches * (1 - mask_ratio))
        mask = torch.zeros(B, N, dtype=torch.bool, device=encoded_features.device)
        for t in range(cutoff, num_temporal_patches):
            s = t * spatial_per_frame; e = (t + 1) * spatial_per_frame
            mask[:, s:e] = True
        masked = encoded_features.clone()
        masked[mask] = 0
        return masked, mask


# ============================================================================
# STAGE 1 TRAINING FUNCTION
# ============================================================================

def train_stage1_droid(
    model, dataloader, num_epochs, device='cuda',
    save_dir='./checkpoints', use_mask_aux=True, lambda_mask=0.1,
    learning_rate=4e-4, weight_decay=0.1
):
    """
    Stage 1: Pre-train encoder on DROID with:
      1) Temporal order classification (shuffled vs. correct)
      2) Symmetric InfoNCE contrastive (orig <-> aug)
      3) (Optional) Structured temporal mask denoising
    """
    model.to(device)
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    temporal_classifier = nn.Sequential(
        nn.Linear(model.encoder_dim, model.encoder_dim // 2),
        nn.ReLU(), nn.Dropout(0.1),
        nn.Linear(model.encoder_dim // 2, 2),
    ).to(device)

    contrastive_proj = nn.Sequential(
        nn.Linear(model.encoder_dim, model.encoder_dim),
        nn.ReLU(),
        nn.Linear(model.encoder_dim, 128),
    ).to(device)

    denoise_head = None
    if use_mask_aux:
        denoise_head = nn.Sequential(
            nn.LayerNorm(model.encoder_dim),
            nn.Linear(model.encoder_dim, model.encoder_dim),
            nn.GELU(),
            nn.Linear(model.encoder_dim, model.encoder_dim),
        ).to(device)

    all_params = (list(model.encoder.parameters())
                  + list(temporal_classifier.parameters())
                  + list(contrastive_proj.parameters())
                  + (list(denoise_head.parameters()) if denoise_head else []))
    optimizer = torch.optim.AdamW(all_params, lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    temperature = 0.07
    best_loss = float("inf")

    print(f"\n{'='*70}")
    print("STAGE 1: Pre-training on DROID")
    print(f"{'='*70}")
    print(f"Epochs: {num_epochs}  |  Batch: {dataloader.batch_size}  |  LR: {learning_rate}")
    print(f"Mask aux: {use_mask_aux}  |  Device: {device}")
    print(f"{'='*70}\n")

    for epoch in range(1, num_epochs + 1):
        model.train(); temporal_classifier.train(); contrastive_proj.train()
        if denoise_head: denoise_head.train()

        ep_temp = ep_ctr = ep_mask = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{num_epochs}")

        for batch in pbar:
            video = batch['video'].to(device)
            video_aug = batch['video_augmented'].to(device)
            shuffled = batch['shuffled_video'].to(device)
            label = batch['order_label'].to(device).long()

            optimizer.zero_grad(set_to_none=True)

            # 1) Temporal order
            shuf_feats = model.encode_video(shuffled).mean(dim=1)
            temporal_loss = F.cross_entropy(temporal_classifier(shuf_feats), label)

            # 2) Contrastive
            f1 = model.encode_video(video).mean(dim=1)
            f2 = model.encode_video(video_aug).mean(dim=1)
            z1 = F.normalize(contrastive_proj(f1), dim=1)
            z2 = F.normalize(contrastive_proj(f2), dim=1)
            labels = torch.arange(z1.size(0), device=device)
            contrastive_loss = 0.5 * (
                F.cross_entropy(z1 @ z2.t() / temperature, labels) +
                F.cross_entropy(z2 @ z1.t() / temperature, labels))

            # 3) Mask aux
            mask_loss = torch.tensor(0.0, device=device)
            if denoise_head is not None:
                with torch.no_grad():
                    enc = model.encode_video(video)
                    T = model.encoder.tubelet_embed.num_temporal_patches
                    masked, mbool = StructuredTemporalMasking.future_masking(enc, T, 0.5)
                um = ~mbool
                denom = um.sum(dim=1, keepdim=True).clamp_min(1)
                pooled = (masked * um.unsqueeze(-1)).sum(dim=1) / denom
                target = enc.mean(dim=1).detach()
                mask_loss = F.mse_loss(denoise_head(pooled), target)

            loss = temporal_loss + contrastive_loss + lambda_mask * mask_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            optimizer.step()

            ep_temp += temporal_loss.item()
            ep_ctr += contrastive_loss.item()
            ep_mask += mask_loss.item()
            pbar.set_postfix(temp=f"{temporal_loss.item():.3f}",
                             ctr=f"{contrastive_loss.item():.3f}",
                             mask=f"{mask_loss.item():.3f}" if denoise_head else "off")

        n = len(dataloader)
        at = ep_temp/max(n,1); ac = ep_ctr/max(n,1); am = ep_mask/max(n,1)
        total = at + ac + (lambda_mask * am if denoise_head else 0)
        scheduler.step()

        print(f"\nEpoch {epoch:02d}: Temporal={at:.4f}  Contrastive={ac:.4f}  "
              f"Mask={am:.4f}  Total={total:.4f}")

        ckpt = {
            'epoch': epoch,
            'encoder_state_dict': model.encoder.state_dict(),
            'temporal_classifier': temporal_classifier.state_dict(),
            'contrastive_proj': contrastive_proj.state_dict(),
            'denoise_head': denoise_head.state_dict() if denoise_head else None,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'loss': total,
        }
        torch.save(ckpt, save_dir / f"checkpoint_epoch_{epoch:02d}.pth")

        if total < best_loss:
            best_loss = total
            torch.save(ckpt, save_dir / "best_encoder_droid.pth")
            print(f"  Saved best checkpoint (loss: {best_loss:.4f})")

    print(f"\n{'='*70}")
    print(f"Stage 1 Complete!  Best loss: {best_loss:.4f}")
    print(f"Checkpoint: {save_dir / 'best_encoder_droid.pth'}")
    print(f"{'='*70}\n")


# ============================================================================
# DATASET FACTORY
# ============================================================================

def create_dataset(data_path, fmt, num_frames, img_size, camera, training,
                   augmentation_strength, max_episodes):
    """Create the appropriate dataset based on format."""
    if fmt == 'rlds':
        return DROIDDatasetRLDS(data_path, num_frames, img_size, camera,
                                training, augmentation_strength, max_episodes)
    elif fmt == 'huggingface':
        hf_cam = f'observation.images.{camera}'
        return DROIDDatasetHuggingFace(data_path, num_frames, img_size, hf_cam,
                                       training, augmentation_strength, max_episodes)
    elif fmt == 'hdf5':
        return DROIDDatasetHDF5(data_path, num_frames, img_size, camera,
                                training, augmentation_strength, max_episodes)
    else:
        raise ValueError(f"Unknown format: {fmt}. Choose rlds, huggingface, or hdf5")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Stage 1: DROID Pre-training')

    # Data
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to DROID data (GCS bucket, HuggingFace ID, or local dir)')
    parser.add_argument('--format', type=str, default='hdf5',
                        choices=['rlds', 'huggingface', 'hdf5'],
                        help='Dataset format')
    parser.add_argument('--camera', type=str, default='exterior_image_1_left',
                        help='Camera view to use')
    parser.add_argument('--max_episodes', type=int, default=None,
                        help='Max episodes to load (for debugging)')
    parser.add_argument('--save_dir', type=str, default='./checkpoints')

    # Model
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--num_frames', type=int, default=16)
    parser.add_argument('--encoder_dim', type=int, default=1024)
    parser.add_argument('--encoder_depth', type=int, default=12)
    parser.add_argument('--encoder_heads', type=int, default=12)
    parser.add_argument('--action_dim', type=int, default=7)
    parser.add_argument('--state_dim', type=int, default=14)

    # Training
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=4e-4)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--augmentation_strength', type=str, default='medium',
                        choices=['light', 'medium', 'heavy'])
    parser.add_argument('--use_mask_aux', action='store_true')
    parser.add_argument('--lambda_mask', type=float, default=0.1)

    # Hardware
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    # Seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Device
    if args.device == 'auto':
        if torch.cuda.is_available():
            args.device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            args.device = 'mps'
        else:
            args.device = 'cpu'
    print(f"Device: {args.device}")

    print(f"\n{'='*70}")
    print("DROID WORLD MODEL — Stage 1 Pre-training")
    print(f"{'='*70}")
    print(f"Data: {args.data_path}  ({args.format})")
    print(f"Camera: {args.camera}")
    print(f"Batch: {args.batch_size}  |  Epochs: {args.num_epochs}  |  LR: {args.learning_rate}")
    print(f"Encoder: dim={args.encoder_dim} depth={args.encoder_depth}")
    print(f"Action dim: {args.action_dim}  |  State dim: {args.state_dim}")
    print(f"{'='*70}\n")

    # Model
    print("Initializing model...")
    model = DROIDActionConditionedWorldModel(
        img_size=args.img_size, num_frames=args.num_frames,
        encoder_dim=args.encoder_dim, encoder_depth=args.encoder_depth,
        encoder_heads=args.encoder_heads,
        predictor_dim=1024, predictor_depth=12, predictor_heads=8,
        action_dim=args.action_dim, state_dim=args.state_dim,
        temporal_scale=1.0, use_grad_checkpoint=True)
    print(f"  {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters\n")

    # Dataset
    print("Loading DROID dataset...")
    dataset = create_dataset(
        args.data_path, args.format, args.num_frames, args.img_size,
        args.camera, True, args.augmentation_strength, args.max_episodes)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True)
    print(f"  {len(dataset)} samples\n")

    # Train
    train_stage1_droid(
        model, dataloader, args.num_epochs, args.device, args.save_dir,
        args.use_mask_aux, args.lambda_mask, args.learning_rate, args.weight_decay)


if __name__ == "__main__":
    main()

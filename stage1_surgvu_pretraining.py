"""
STAGE 1: SurgVU Pre-training (Standalone Script)
================================================

This script handles pre-training the video encoder on the SurgVU dataset using:
1. Temporal order prediction (shuffled vs. correct sequence)
2. Symmetric InfoNCE contrastive learning
3. Optional: Structured temporal mask denoising

Usage:
    python stage1_surgvu_pretraining.py --data_root ./data/SurgVU --save_dir ./checkpoints
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

# Import the model (assumes surgical_world_model.py is in the same directory)
from surgical_world_model import SurgicalActionConditionedWorldModel


# ============================================================================
# SURGVU DATASET IMPLEMENTATION
# ============================================================================

class SurgVUDataset(Dataset):
    """
    Complete SurgVU dataset implementation for self-supervised pre-training.
    
    Dataset Structure:
    - 280 long videos from da Vinci robotic surgery training
    - 8 task categories (suturing, dissection, vessel isolation, etc.)
    - 60 FPS, 720p resolution
    - ~840 hours total
    
    Pre-training Tasks:
    1. Temporal order prediction (shuffle frames, predict correct order)
    2. Contrastive learning (temporally close frames = positive pairs)
    """
    
    def __init__(
        self,
        root,
        num_frames=16,
        img_size=224,
        sample_fps=1,  # Downsample from 60 FPS to 1 FPS for efficiency
        training=True,
        augmentation_strength='medium'
    ):
        self.root = Path(root)
        self.num_frames = num_frames
        self.img_size = img_size
        self.sample_fps = sample_fps
        self.training = training
        
        # Load video list
        self.video_files = self._discover_videos()
        
        # Compute frame indices for each video
        self.samples = self._create_samples()
        
        # Augmentation
        self.augmentation = SurgicalVideoAugmentation(
            img_size, training, strength=augmentation_strength
        )
        
        print(f"SurgVU Dataset: {len(self.samples)} clips from {len(self.video_files)} videos")
    
    def _discover_videos(self):
        """Discover all video files in the dataset."""
        video_files = []
        video_dir = self.root / 'videos'
        
        if not video_dir.exists():
            raise FileNotFoundError(f"Video directory not found: {video_dir}")
        
        for video_file in video_dir.glob('*.mp4'):
            video_files.append(video_file)
        
        return sorted(video_files)
    
    def _create_samples(self):
        """
        Create training samples from videos.
        
        Strategy: Sample at 1 FPS with 16-frame clips, stride 8 frames
        to have 50% overlap for more training data.
        """
        samples = []
        
        for video_file in self.video_files:
            # Get video info
            cap = cv2.VideoCapture(str(video_file))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            cap.release()
            
            # Calculate sampling stride
            frame_stride = fps // self.sample_fps
            
            # Create clips with stride
            for start_frame in range(0, total_frames, frame_stride * 8):
                end_frame = start_frame + frame_stride * self.num_frames
                
                if end_frame <= total_frames:
                    samples.append({
                        'video_file': video_file,
                        'start_frame': start_frame,
                        'frame_stride': frame_stride,
                        'num_frames': self.num_frames
                    })
        
        return samples
    
    def _load_video_clip(self, video_file, start_frame, frame_stride, num_frames):
        """Load a video clip with specified sampling."""
        frames = []
        cap = cv2.VideoCapture(str(video_file))
        
        for i in range(num_frames):
            frame_idx = start_frame + i * frame_stride
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            else:
                # If frame reading fails, duplicate last frame
                frames.append(frames[-1] if frames else np.zeros((720, 1280, 3), dtype=np.uint8))
        
        cap.release()
        return frames
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load video clip
        frames = self._load_video_clip(
            sample['video_file'],
            sample['start_frame'],
            sample['frame_stride'],
            sample['num_frames']
        )
        
        # Apply augmentation
        video = self.augmentation(frames)  # (C, T, H, W)
        
        if self.training:
            # Create temporal order prediction task
            # Shuffle frames and create label for correct order
            shuffled_video, order_label = self._create_temporal_order_task(video)
            
            # Create contrastive pair (different augmentation of same clip)
            video_augmented = self.augmentation(frames)
            
            return {
                'video': video,
                'video_augmented': video_augmented,
                'shuffled_video': shuffled_video,
                'order_label': order_label
            }
        else:
            return {'video': video}
    
    def _create_temporal_order_task(self, video):
        """
        Create temporal order prediction task.
        
        Strategy: Randomly shuffle frames and the model must predict
        if the sequence is in correct temporal order or not.
        
        Returns:
            shuffled_video: Potentially shuffled frames
            label: 0 = correct order, 1 = shuffled
        """
        if np.random.random() < 0.5:
            # Keep correct order
            return video, 0
        else:
            # Shuffle frames
            C, T, H, W = video.shape
            perm = torch.randperm(T)
            shuffled = video[:, perm, :, :]
            return shuffled, 1


class SurgicalVideoAugmentation:
    """
    Enhanced data augmentation for surgical videos with adjustable strength.
    
    Augmentation Techniques:
    - Spatial: Random crop, rotation, color jitter
    - Temporal: Random frame sampling, temporal jittering
    - Avoid: Vertical flip (breaks anatomical consistency)
    """
    
    def __init__(self, img_size=224, training=True, strength='medium'):
        self.img_size = img_size
        self.training = training
        self.strength = strength
        
        if training:
            if strength == 'light':
                scale_range = (0.9, 1.0)
                rotation_degrees = 5
                brightness = 0.1
                contrast = 0.1
            elif strength == 'medium':
                scale_range = (0.8, 1.0)
                rotation_degrees = 10
                brightness = 0.2
                contrast = 0.15
            else:  # 'heavy' for few-shot learning
                scale_range = (0.7, 1.0)
                rotation_degrees = 15
                brightness = 0.3
                contrast = 0.2
            
            self.spatial_transform = transforms.Compose([
                transforms.RandomResizedCrop(img_size, scale=scale_range),
                transforms.RandomRotation(degrees=rotation_degrees),
                transforms.ColorJitter(brightness=brightness, contrast=contrast),
            ])
        else:
            self.spatial_transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.CenterCrop(img_size),
            ])
        
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    
    def __call__(self, video_frames):
        """
        Args:
            video_frames: List of PIL Images or numpy arrays
        Returns:
            augmented: Tensor (C, T, H, W)
        """
        augmented_frames = []
        
        # Apply same spatial transformation to all frames for temporal consistency
        if self.training:
            # Get random parameters for the transformation
            transform_params = self.spatial_transform.transforms[0].get_params(
                transforms.ToPILImage()(video_frames[0]) if isinstance(video_frames[0], np.ndarray) else video_frames[0],
                self.spatial_transform.transforms[0].scale,
                self.spatial_transform.transforms[0].ratio
            )
        
        for frame in video_frames:
            # Convert to PIL if numpy
            if isinstance(frame, np.ndarray):
                frame = transforms.ToPILImage()(frame)
            
            if self.training:
                # Apply same crop to maintain consistency
                frame = transforms.functional.resized_crop(frame, *transform_params, (self.img_size, self.img_size))
                frame = self.spatial_transform.transforms[1](frame)  # Rotation
                frame = self.spatial_transform.transforms[2](frame)  # Color jitter
            else:
                frame = self.spatial_transform(frame)
            
            frame = transforms.ToTensor()(frame)
            frame = self.normalize(frame)
            augmented_frames.append(frame)
        
        # Stack: (T, C, H, W) -> (C, T, H, W)
        video = torch.stack(augmented_frames, dim=0)
        video = video.permute(1, 0, 2, 3)
        
        return video


# ============================================================================
# STRUCTURED TEMPORAL MASKING (OPTIONAL)
# ============================================================================

class StructuredTemporalMasking:
    """
    Structured temporal masking for pre-training.
    
    Instead of random token masking, we mask entire future timesteps.
    This is more suitable for video and encourages temporal prediction.
    """
    
    @staticmethod
    def future_masking(encoded_features, num_temporal_patches, mask_ratio=0.5):
        """
        Mask all future timesteps after a random cutoff point.
        
        Args:
            encoded_features: (B, N, D) where N = T * spatial_patches
            num_temporal_patches: T
            mask_ratio: Fraction of temporal dimension to mask
            
        Returns:
            masked_features: Same shape with future masked to zero
            mask: (B, N) boolean mask (True = masked)
        """
        B, N, D = encoded_features.shape
        spatial_patches_per_frame = N // num_temporal_patches
        
        # Choose random cutoff point
        cutoff = int(num_temporal_patches * (1 - mask_ratio))
        
        # Create mask
        mask = torch.zeros(B, N, dtype=torch.bool, device=encoded_features.device)
        
        # Mask all patches after cutoff
        for t in range(cutoff, num_temporal_patches):
            start_idx = t * spatial_patches_per_frame
            end_idx = (t + 1) * spatial_patches_per_frame
            mask[:, start_idx:end_idx] = True
        
        # Apply mask
        masked_features = encoded_features.clone()
        masked_features[mask] = 0
        
        return masked_features, mask


# ============================================================================
# STAGE 1 TRAINING FUNCTION
# ============================================================================

def train_stage1_surgvu(
    model,
    dataloader,
    num_epochs,
    device='cuda',
    save_dir='./checkpoints',
    use_mask_aux=True,
    lambda_mask=0.1,
    learning_rate=4e-4,
    weight_decay=0.1
):
    """
    Stage 1: Pre-train encoder on SurgVU with:
      1) Temporal order classification (shuffled vs. correct)
      2) Symmetric InfoNCE contrastive (orig ↔ aug)
      3) (Optional) Structured temporal mask denoising auxiliary loss in latent space
    """
    model.to(device)
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    # Heads on top of encoder features
    temporal_classifier = nn.Sequential(
        nn.Linear(model.encoder_dim, model.encoder_dim // 2),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(model.encoder_dim // 2, 2),  # correct vs shuffled
    ).to(device)

    contrastive_proj = nn.Sequential(
        nn.Linear(model.encoder_dim, model.encoder_dim),
        nn.ReLU(),
        nn.Linear(model.encoder_dim, 128),
    ).to(device)

    # Optional latent denoising head for auxiliary loss
    denoise_head = None
    if use_mask_aux:
        denoise_head = nn.Sequential(
            nn.LayerNorm(model.encoder_dim),
            nn.Linear(model.encoder_dim, model.encoder_dim),
            nn.GELU(),
            nn.Linear(model.encoder_dim, model.encoder_dim),
        ).to(device)

    # Optimizer / scheduler
    all_params = list(model.encoder.parameters()) \
               + list(temporal_classifier.parameters()) \
               + list(contrastive_proj.parameters()) \
               + (list(denoise_head.parameters()) if denoise_head is not None else [])
    optimizer = torch.optim.AdamW(all_params, lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    temperature = 0.07
    best_loss = float("inf")

    print(f"\n{'='*70}")
    print("STAGE 1: Pre-training on SurgVU")
    print(f"{'='*70}")
    print(f"Number of epochs: {num_epochs}")
    print(f"Batch size: {dataloader.batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Mask auxiliary loss: {use_mask_aux}")
    print(f"{'='*70}\n")

    for epoch in range(1, num_epochs + 1):
        model.train()
        temporal_classifier.train()
        contrastive_proj.train()
        if denoise_head is not None:
            denoise_head.train()

        epoch_temporal_loss = 0.0
        epoch_contrastive_loss = 0.0
        epoch_mask_loss = 0.0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{num_epochs}")
        for batch in pbar:
            video = batch["video"].to(device)                   # (B, C, T, H, W)
            video_aug = batch["video_augmented"].to(device)     # (B, C, T, H, W)
            shuffled_video = batch["shuffled_video"].to(device) # (B, C, T, H, W)
            order_label = batch["order_label"].to(device).long()

            optimizer.zero_grad(set_to_none=True)

            # === 1) Temporal order classification ===
            shuffled_feats = model.encode_video(shuffled_video).mean(dim=1)  # (B, D)
            order_logits = temporal_classifier(shuffled_feats)
            temporal_loss = F.cross_entropy(order_logits, order_label)

            # === 2) Symmetric InfoNCE (orig ↔ aug) ===
            f_orig = model.encode_video(video).mean(dim=1)       # (B, D)
            f_aug  = model.encode_video(video_aug).mean(dim=1)   # (B, D)

            z1 = F.normalize(contrastive_proj(f_orig), dim=1)    # (B, 128)
            z2 = F.normalize(contrastive_proj(f_aug),  dim=1)    # (B, 128)

            # logits[i, j] = sim(z1[i], z2[j]) / tau
            logits_12 = (z1 @ z2.t()) / temperature              # (B, B)
            logits_21 = (z2 @ z1.t()) / temperature              # (B, B)
            labels = torch.arange(z1.size(0), device=device)

            loss_12 = F.cross_entropy(logits_12, labels)
            loss_21 = F.cross_entropy(logits_21, labels)
            contrastive_loss = 0.5 * (loss_12 + loss_21)

            # === 3) Optional: structured temporal mask denoising auxiliary ===
            mask_loss = torch.tensor(0.0, device=device)
            if denoise_head is not None:
                with torch.no_grad():
                    enc_tokens = model.encode_video(video)                      # (B, N, Denc)
                    T = model.encoder.tubelet_embed.num_temporal_patches
                    masked_tokens, mask_bool = StructuredTemporalMasking.future_masking(
                        enc_tokens, num_temporal_patches=T, mask_ratio=0.5
                    )

                # Pool only UNMASKED tokens
                unmasked = ~mask_bool                                          # (B, N)
                denom = unmasked.sum(dim=1, keepdim=True).clamp_min(1)         # (B, 1)
                pooled_masked = (masked_tokens * unmasked.unsqueeze(-1)).sum(dim=1) / denom
                target_pooled = enc_tokens.mean(dim=1).detach()                # (B, Denc)

                mask_loss = F.mse_loss(denoise_head(pooled_masked), target_pooled)

            # === Combined loss ===
            loss = temporal_loss + contrastive_loss + (lambda_mask * mask_loss)

            # Backprop
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            optimizer.step()

            epoch_temporal_loss += float(temporal_loss.detach())
            epoch_contrastive_loss += float(contrastive_loss.detach())
            epoch_mask_loss += float(mask_loss.detach())

            pbar.set_postfix({
                "temp": f"{temporal_loss.item():.3f}",
                "ctr": f"{contrastive_loss.item():.3f}",
                "mask": f"{mask_loss.item():.3f}" if denoise_head is not None else "off",
            })

        # Epoch metrics
        n = len(dataloader)
        avg_temporal = epoch_temporal_loss / max(n, 1)
        avg_contrastive = epoch_contrastive_loss / max(n, 1)
        avg_mask = epoch_mask_loss / max(n, 1)
        avg_total = avg_temporal + avg_contrastive + (lambda_mask * avg_mask if denoise_head is not None else 0.0)

        scheduler.step()

        print(
            f"\nEpoch {epoch:02d} Summary:\n"
            f"  Temporal Loss: {avg_temporal:.4f}\n"
            f"  Contrastive Loss: {avg_contrastive:.4f}\n"
            f"  Mask Aux Loss: {avg_mask:.4f}\n"
            f"  Total Loss: {avg_total:.4f}"
        )
        epoch_checkpoint = {
            "epoch": epoch,
            "encoder_state_dict": model.encoder.state_dict(),
            "temporal_classifier": temporal_classifier.state_dict(),
            "contrastive_proj": contrastive_proj.state_dict(),
            "denoise_head": (denoise_head.state_dict() if denoise_head is not None else None),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "loss": avg_total,
        }
        torch.save(epoch_checkpoint, save_dir / f"checkpoint_epoch_{epoch:02d}.pth")
        print(f"  ✓ Saved epoch {epoch} checkpoint")

        # Save best encoder checkpoint
        if avg_total < best_loss:
            best_loss = avg_total
            checkpoint = {
                "epoch": epoch,
                "encoder_state_dict": model.encoder.state_dict(),
                "temporal_classifier": temporal_classifier.state_dict(),
                "contrastive_proj": contrastive_proj.state_dict(),
                "denoise_head": (denoise_head.state_dict() if denoise_head is not None else None),
                "loss": best_loss,
            }
            torch.save(checkpoint, save_dir / "best_encoder_surgvu.pth")
            print(f"  ✓ Saved best checkpoint (loss: {best_loss:.4f})")

    print(f"\n{'='*70}")
    print(f"Stage 1 Complete!")
    print(f"{'='*70}")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Checkpoint saved to: {save_dir / 'best_encoder_surgvu.pth'}")
    print(f"{'='*70}\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Stage 1: SurgVU Pre-training')
    
    # Data arguments
    parser.add_argument('--data_root', type=str, required=True,
                        help='Path to SurgVU dataset root directory')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    
    # Model arguments
    parser.add_argument('--img_size', type=int, default=224,
                        help='Image size')
    parser.add_argument('--num_frames', type=int, default=16,
                        help='Number of frames per clip')
    parser.add_argument('--encoder_dim', type=int, default=1024,
                        help='Encoder embedding dimension')
    parser.add_argument('--encoder_depth', type=int, default=12,
                        help='Number of encoder layers')
    
    # Training arguments
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=4e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.1,
                        help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of dataloader workers')
    
    # Loss arguments
    parser.add_argument('--use_mask_aux', action='store_true',
                        help='Use mask denoising auxiliary loss')
    parser.add_argument('--lambda_mask', type=float, default=0.1,
                        help='Weight for mask auxiliary loss')
    
    # Hardware arguments
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto/cuda/mps/cpu)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # MPS doesn't need manual seed setting
        pass
    
    # Auto-detect best available device
    if args.device == 'auto':
        if torch.cuda.is_available():
            args.device = 'cuda'
            print("Auto-detected device: CUDA")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            args.device = 'mps'
            print("Auto-detected device: MPS (Apple Silicon)")
        else:
            args.device = 'cpu'
            print("Auto-detected device: CPU")
    else:
        # Check device availability
        if args.device == 'cuda' and not torch.cuda.is_available():
            print("Warning: CUDA not available, falling back to CPU")
            args.device = 'cpu'
        elif args.device == 'mps':
            if not hasattr(torch.backends, 'mps'):
                print("Warning: MPS not supported in this PyTorch version, falling back to CPU")
                args.device = 'cpu'
            elif not torch.backends.mps.is_available():
                print("Warning: MPS not available, falling back to CPU")
                args.device = 'cpu'
    
    print(f"\n{'='*70}")
    print("SURGICAL WORLD MODEL - Stage 1 Pre-training")
    print(f"{'='*70}")
    print(f"Configuration:")
    print(f"  Data root: {args.data_root}")
    print(f"  Save directory: {args.save_dir}")
    print(f"  Device: {args.device}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.num_epochs}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Encoder dim: {args.encoder_dim}")
    print(f"  Encoder depth: {args.encoder_depth}")
    print(f"{'='*70}\n")
    
    # Initialize model
    print("Initializing model...")
    model = SurgicalActionConditionedWorldModel(
        img_size=args.img_size,
        num_frames=args.num_frames,
        encoder_dim=args.encoder_dim,
        encoder_depth=args.encoder_depth,
        encoder_heads=12,
        predictor_dim=1024,
        predictor_depth=12,
        predictor_heads=8,
        use_grad_checkpoint=True
    )
    
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"✓ Model initialized: {num_params:.1f}M parameters\n")
    
    # Load dataset
    print("Loading SurgVU dataset...")
    dataset = SurgVUDataset(
        root=args.data_root,
        num_frames=args.num_frames,
        img_size=args.img_size,
        sample_fps=1,
        training=True,
        augmentation_strength='medium'
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    print(f"✓ Dataset loaded: {len(dataset)} samples\n")
    
    # Train
    train_stage1_surgvu(
        model=model,
        dataloader=dataloader,
        num_epochs=args.num_epochs,
        device=args.device,
        save_dir=args.save_dir,
        use_mask_aux=args.use_mask_aux,
        lambda_mask=args.lambda_mask,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay
    )


if __name__ == "__main__":
    main()

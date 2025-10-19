"""
STAGE 1: SurgVU Pre-training with V-JEPA 2 Pretrained Weights
===============================================================

This script performs Stage 1 pre-training starting from V-JEPA 2 pretrained
encoder instead of training from scratch. This significantly improves:

1. Convergence speed (faster training)
2. Final performance (better representations)
3. Data efficiency (less data needed)

Usage with V-JEPA 2:
    # First, load V-JEPA 2 weights:
    python load_vjepa2_pretrained.py --model_size vitl --save_path ./checkpoints/vjepa2_encoder.pth
    
    # Then run training:
    python stage1_vjepa2_finetuning.py \
        --data_root ./data/SurgVU \
        --pretrained_checkpoint ./checkpoints/vjepa2_encoder.pth \
        --encoder_frozen  # Freeze encoder, only train predictor

Usage from scratch (original):
    python stage1_vjepa2_finetuning.py \
        --data_root ./data/SurgVU \
        --no_pretrained  # Train from scratch
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

# Import the model and dataset
from surgical_world_model import SurgicalActionConditionedWorldModel

# Try to import from recursive version first, fallback to original
try:
    from stage1_surgvu_pretraining_recursive import (
        SurgVUDataset, 
        SurgicalVideoAugmentation,
        StructuredTemporalMasking
    )
except ImportError:
    from stage1_surgvu_pretraining import (
        SurgVUDataset, 
        SurgicalVideoAugmentation,
        StructuredTemporalMasking
    )


def load_pretrained_model(
    checkpoint_path,
    img_size=224,
    num_frames=16,
    predictor_dim=1024,
    predictor_depth=12,
    predictor_heads=8,
    device='cuda'
):
    """
    Load surgical model with V-JEPA 2 pretrained encoder.
    
    Args:
        checkpoint_path: Path to V-JEPA 2 adapted checkpoint
        img_size: Input image size
        num_frames: Number of frames
        predictor_dim: Predictor dimension
        predictor_depth: Predictor depth
        predictor_heads: Predictor heads
        device: Device to load model on
        
    Returns:
        model: Loaded model
        config: Model configuration
    """
    print(f"\n{'='*70}")
    print("LOADING PRETRAINED V-JEPA 2 MODEL")
    print(f"{'='*70}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Get V-JEPA 2 configuration
    vjepa2_config = checkpoint.get('vjepa2_config', {})
    encoder_dim = vjepa2_config.get('embed_dim', 1024)
    encoder_depth = vjepa2_config.get('depth', 24)
    encoder_heads = vjepa2_config.get('num_heads', 16)
    
    print(f"\nV-JEPA 2 Configuration:")
    print(f"  - Encoder dim: {encoder_dim}")
    print(f"  - Encoder depth: {encoder_depth}")
    print(f"  - Encoder heads: {encoder_heads}")
    
    # Create model with V-JEPA 2 dimensions
    model = SurgicalActionConditionedWorldModel(
        img_size=img_size,
        num_frames=num_frames,
        tubelet_size=(2, 16, 16),
        in_channels=3,
        encoder_dim=encoder_dim,
        encoder_depth=encoder_depth,
        encoder_heads=encoder_heads,
        predictor_dim=predictor_dim,
        predictor_depth=predictor_depth,
        predictor_heads=predictor_heads,
        action_dim=7,
        state_dim=7,
        num_anomaly_classes=2,
        use_grad_checkpoint=True
    )
    
    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print("✓ Loaded full model state dict")
    elif 'encoder_state_dict' in checkpoint:
        model.encoder.load_state_dict(checkpoint['encoder_state_dict'], strict=False)
        print("✓ Loaded encoder state dict")
    else:
        raise ValueError("Checkpoint does not contain model weights")
    
    model = model.to(device)
    
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"✓ Model loaded: {num_params:.1f}M parameters")
    
    config = {
        'encoder_dim': encoder_dim,
        'encoder_depth': encoder_depth,
        'encoder_heads': encoder_heads
    }
    
    return model, config


def train_stage1_with_vjepa2(
    model,
    dataloader,
    num_epochs,
    device='cuda',
    save_dir='./checkpoints',
    use_mask_aux=True,
    lambda_mask=0.1,
    learning_rate=4e-4,
    weight_decay=0.1,
    encoder_frozen=True,
    warmup_epochs=2
):
    """
    Stage 1 training with V-JEPA 2 pretrained encoder.
    
    Key differences from training from scratch:
    1. Lower learning rate (encoder already has good features)
    2. Optional encoder freezing (faster, more stable)
    3. Warmup period for predictor to adapt
    4. Focus on contrastive + temporal tasks
    """
    model.to(device)
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    # Freeze encoder if requested
    if encoder_frozen:
        print(f"\n✓ Freezing V-JEPA 2 encoder (only predictor will be trained)")
        for param in model.encoder.parameters():
            param.requires_grad = False
    else:
        print(f"\n✓ Encoder unfrozen (will be fine-tuned)")
    
    # Task-specific heads
    temporal_classifier = nn.Sequential(
        nn.Linear(model.encoder_dim, model.encoder_dim // 2),
        nn.ReLU(),
        nn.Dropout(0.1),
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
    
    # Optimizer setup - different learning rates for encoder vs. heads
    if encoder_frozen:
        # Only train predictor and heads
        trainable_params = [
            {'params': model.action_state_embed.parameters(), 'lr': learning_rate},
            {'params': model.video_proj.parameters(), 'lr': learning_rate},
            {'params': model.predictor_blocks.parameters(), 'lr': learning_rate},
            {'params': model.predictor_norm.parameters(), 'lr': learning_rate},
            {'params': model.predictor_to_encoder.parameters(), 'lr': learning_rate},
            {'params': model.sterility_classifier.parameters(), 'lr': learning_rate},
            {'params': model.autoregressive_predictor.parameters(), 'lr': learning_rate},
            {'params': temporal_classifier.parameters(), 'lr': learning_rate},
            {'params': contrastive_proj.parameters(), 'lr': learning_rate},
        ]
        if denoise_head is not None:
            trainable_params.append({'params': denoise_head.parameters(), 'lr': learning_rate})
    else:
        # Fine-tune encoder with lower learning rate
        encoder_lr = learning_rate * 0.1  # 10x lower for encoder
        trainable_params = [
            {'params': model.encoder.parameters(), 'lr': encoder_lr},
            {'params': model.action_state_embed.parameters(), 'lr': learning_rate},
            {'params': model.video_proj.parameters(), 'lr': learning_rate},
            {'params': model.predictor_blocks.parameters(), 'lr': learning_rate},
            {'params': model.predictor_norm.parameters(), 'lr': learning_rate},
            {'params': model.predictor_to_encoder.parameters(), 'lr': learning_rate},
            {'params': model.sterility_classifier.parameters(), 'lr': learning_rate},
            {'params': model.autoregressive_predictor.parameters(), 'lr': learning_rate},
            {'params': temporal_classifier.parameters(), 'lr': learning_rate},
            {'params': contrastive_proj.parameters(), 'lr': learning_rate},
        ]
        if denoise_head is not None:
            trainable_params.append({'params': denoise_head.parameters(), 'lr': learning_rate})
    
    optimizer = torch.optim.AdamW(trainable_params, weight_decay=weight_decay)
    
    # Scheduler with warmup
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            return 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (num_epochs - warmup_epochs)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    temperature = 0.07
    best_loss = float("inf")
    
    print(f"\n{'='*70}")
    print("STAGE 1: Fine-tuning on SurgVU with V-JEPA 2")
    print(f"{'='*70}")
    print(f"Number of epochs: {num_epochs}")
    print(f"Batch size: {dataloader.batch_size}")
    print(f"Base learning rate: {learning_rate}")
    if not encoder_frozen:
        print(f"Encoder learning rate: {encoder_lr}")
    print(f"Warmup epochs: {warmup_epochs}")
    print(f"Encoder frozen: {encoder_frozen}")
    print(f"Mask auxiliary loss: {use_mask_aux}")
    
    # Count trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    total = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Trainable parameters: {trainable:.1f}M / {total:.1f}M ({100*trainable/total:.1f}%)")
    print(f"{'='*70}\n")
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        temporal_classifier.train()
        contrastive_proj.train()
        if denoise_head is not None:
            denoise_head.train()
        
        # Set encoder to eval mode if frozen (for batchnorm, dropout)
        if encoder_frozen:
            model.encoder.eval()
        
        epoch_temporal_loss = 0.0
        epoch_contrastive_loss = 0.0
        epoch_mask_loss = 0.0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{num_epochs}")
        for batch_idx, batch in enumerate(pbar):
            video = batch["video"].to(device)
            video_aug = batch["video_augmented"].to(device)
            shuffled_video = batch["shuffled_video"].to(device)
            order_label = batch["order_label"].to(device).long()
            
            optimizer.zero_grad(set_to_none=True)
            
            # 1) Temporal order classification
            with torch.set_grad_enabled(not encoder_frozen):
                shuffled_feats = model.encode_video(shuffled_video).mean(dim=1)
            order_logits = temporal_classifier(shuffled_feats)
            temporal_loss = F.cross_entropy(order_logits, order_label)
            
            # 2) Symmetric InfoNCE contrastive
            with torch.set_grad_enabled(not encoder_frozen):
                f_orig = model.encode_video(video).mean(dim=1)
                f_aug = model.encode_video(video_aug).mean(dim=1)
            
            z1 = F.normalize(contrastive_proj(f_orig), dim=1)
            z2 = F.normalize(contrastive_proj(f_aug), dim=1)
            
            logits_12 = (z1 @ z2.t()) / temperature
            logits_21 = (z2 @ z1.t()) / temperature
            labels = torch.arange(z1.size(0), device=device)
            
            loss_12 = F.cross_entropy(logits_12, labels)
            loss_21 = F.cross_entropy(logits_21, labels)
            contrastive_loss = 0.5 * (loss_12 + loss_21)
            
            # 3) Optional mask denoising auxiliary
            mask_loss = torch.tensor(0.0, device=device)
            if denoise_head is not None:
                with torch.no_grad():
                    enc_tokens = model.encode_video(video)
                    T = model.encoder.tubelet_embed.num_temporal_patches
                    masked_tokens, mask_bool = StructuredTemporalMasking.future_masking(
                        enc_tokens, num_temporal_patches=T, mask_ratio=0.5
                    )
                
                unmasked = ~mask_bool
                denom = unmasked.sum(dim=1, keepdim=True).clamp_min(1)
                pooled_masked = (masked_tokens * unmasked.unsqueeze(-1)).sum(dim=1) / denom
                target_pooled = enc_tokens.mean(dim=1).detach()
                
                mask_loss = F.mse_loss(denoise_head(pooled_masked), target_pooled)
            
            # Combined loss
            loss = temporal_loss + contrastive_loss + (lambda_mask * mask_loss)
            
            # Backprop
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad] + 
                list(temporal_classifier.parameters()) +
                list(contrastive_proj.parameters()) +
                (list(denoise_head.parameters()) if denoise_head is not None else []),
                max_norm=1.0
            )
            optimizer.step()
            
            epoch_temporal_loss += float(temporal_loss.detach())
            epoch_contrastive_loss += float(contrastive_loss.detach())
            epoch_mask_loss += float(mask_loss.detach())
            
            pbar.set_postfix({
                "temp": f"{temporal_loss.item():.3f}",
                "ctr": f"{contrastive_loss.item():.3f}",
                "mask": f"{mask_loss.item():.3f}" if denoise_head is not None else "off",
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}"
            })
        
        # Epoch summary
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
            f"  Total Loss: {avg_total:.4f}\n"
            f"  Learning Rate: {optimizer.param_groups[0]['lr']:.2e}"
        )
        
        # Save best checkpoint
        if avg_total < best_loss:
            best_loss = avg_total
            checkpoint = {
                "epoch": epoch,
                "encoder_state_dict": model.encoder.state_dict(),
                "full_model_state_dict": model.state_dict(),
                "temporal_classifier": temporal_classifier.state_dict(),
                "contrastive_proj": contrastive_proj.state_dict(),
                "denoise_head": (denoise_head.state_dict() if denoise_head is not None else None),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "loss": best_loss,
                "config": {
                    'encoder_frozen': encoder_frozen,
                    'encoder_dim': model.encoder_dim,
                    'pretrained_source': 'V-JEPA 2'
                }
            }
            torch.save(checkpoint, save_dir / "best_vjepa2_finetuned.pth")
            print(f"  ✓ Saved best checkpoint (loss: {best_loss:.4f})")
    
    print(f"\n{'='*70}")
    print(f"Stage 1 Fine-tuning Complete!")
    print(f"{'='*70}")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Checkpoint saved to: {save_dir / 'best_vjepa2_finetuned.pth'}")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Stage 1: Fine-tune V-JEPA 2 on SurgVU'
    )
    
    # Data arguments
    parser.add_argument('--data_root', type=str, required=True,
                        help='Path to SurgVU dataset root directory')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    
    # Pretrained model arguments
    parser.add_argument('--pretrained_checkpoint', type=str, default=None,
                        help='Path to V-JEPA 2 adapted checkpoint')
    parser.add_argument('--no_pretrained', action='store_true',
                        help='Train from scratch (not recommended)')
    
    # Model arguments
    parser.add_argument('--img_size', type=int, default=224,
                        help='Image size')
    parser.add_argument('--num_frames', type=int, default=16,
                        help='Number of frames per clip')
    parser.add_argument('--predictor_dim', type=int, default=1024,
                        help='Predictor embedding dimension')
    parser.add_argument('--predictor_depth', type=int, default=12,
                        help='Number of predictor layers')
    
    # Training arguments
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-4,
                        help='Learning rate (lower than from-scratch)')
    parser.add_argument('--weight_decay', type=float, default=0.1,
                        help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of dataloader workers')
    parser.add_argument('--warmup_epochs', type=int, default=2,
                        help='Warmup epochs')
    
    # Encoder arguments
    parser.add_argument('--encoder_frozen', action='store_true', default=True,
                        help='Freeze encoder (recommended)')
    parser.add_argument('--finetune_encoder', action='store_false', dest='encoder_frozen',
                        help='Fine-tune encoder (slower, more GPU memory)')
    
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
    
    # Auto-detect device
    if args.device == 'auto':
        if torch.cuda.is_available():
            args.device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            args.device = 'mps'
        else:
            args.device = 'cpu'
    
    print(f"\n{'='*70}")
    print("STAGE 1: SURGICAL WORLD MODEL WITH V-JEPA 2")
    print(f"{'='*70}")
    print(f"Configuration:")
    print(f"  Data root: {args.data_root}")
    print(f"  Pretrained checkpoint: {args.pretrained_checkpoint or 'None (from scratch)'}")
    print(f"  Save directory: {args.save_dir}")
    print(f"  Device: {args.device}")
    print(f"  Encoder frozen: {args.encoder_frozen}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.num_epochs}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"{'='*70}\n")
    
    # Check if pretrained checkpoint exists
    if not args.no_pretrained and args.pretrained_checkpoint is None:
        print("⚠ No pretrained checkpoint specified!")
        print("\nTo use V-JEPA 2 pretrained weights:")
        print("  1. Run: python load_vjepa2_pretrained.py")
        print("  2. Then: python stage1_vjepa2_finetuning.py --pretrained_checkpoint <path>\n")
        print("Or use --no_pretrained to train from scratch (not recommended)\n")
        return
    
    # Load model
    if args.no_pretrained or args.pretrained_checkpoint is None:
        print("⚠ Training from scratch (not recommended - V-JEPA 2 is much better!)")
        from stage1_surgvu_pretraining import main as train_from_scratch
        train_from_scratch()
        return
    
    # Load V-JEPA 2 pretrained model
    print("Loading V-JEPA 2 pretrained model...")
    model, config = load_pretrained_model(
        args.pretrained_checkpoint,
        img_size=args.img_size,
        num_frames=args.num_frames,
        predictor_dim=args.predictor_dim,
        predictor_depth=args.predictor_depth,
        predictor_heads=8,
        device=args.device
    )
    
    # Load dataset
    print("\nLoading SurgVU dataset...")
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
    train_stage1_with_vjepa2(
        model=model,
        dataloader=dataloader,
        num_epochs=args.num_epochs,
        device=args.device,
        save_dir=args.save_dir,
        use_mask_aux=args.use_mask_aux,
        lambda_mask=args.lambda_mask,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        encoder_frozen=args.encoder_frozen,
        warmup_epochs=args.warmup_epochs
    )


if __name__ == "__main__":
    main()

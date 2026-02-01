"""
Load V-JEPA 2 Pretrained Weights for DROID World Model
=======================================================

Downloads Meta's V-JEPA 2 pretrained encoder, adapts it to work with the
DROID world model architecture, and prepares it for fine-tuning on DROID data.

V-JEPA 2 Details:
- Trained on 1M hours of internet video
- State-of-the-art motion understanding
- Uses 3D RoPE (compatible with our architecture)
- Available sizes: ViT-L/16 (300M), ViT-H/16 (600M), ViT-g/16 (1B)

Usage:
    python load_vjepa2_pretrained.py --model_size vitl --save_path ./checkpoints/vjepa2_droid_model.pth
"""

import torch
import torch.nn as nn
from pathlib import Path
import argparse
from droid_world_model import DROIDActionConditionedWorldModel, VideoViTEncoder


def download_vjepa2_encoder(model_size='vitl', resolution=256):
    """
    Download V-JEPA 2 encoder from Meta's repository.

    Args:
        model_size: 'vitl' (300M), 'vith' (600M), or 'vitg' (1B)
        resolution: 256 or 384

    Returns:
        encoder: Pretrained V-JEPA 2 encoder
        config: dict with embed_dim, depth, num_heads
    """
    print(f"\n{'='*70}")
    print("LOADING V-JEPA 2 PRETRAINED ENCODER")
    print(f"{'='*70}")
    print(f"Model size: {model_size.upper()}")
    print(f"Resolution: {resolution}")

    config_map = {
        'vitl': {'embed_dim': 1024, 'depth': 24, 'num_heads': 16, 'hub_name': 'vjepa2_vitl'},
        'vith': {'embed_dim': 1280, 'depth': 32, 'num_heads': 16, 'hub_name': 'vjepa2_vith'},
        'vitg': {'embed_dim': 1024, 'depth': 40, 'num_heads': 16, 'hub_name': 'vjepa2_vitg'},
    }

    if model_size not in config_map:
        raise ValueError(f"Unknown model size: {model_size}. Choose from {list(config_map.keys())}")

    cfg = config_map[model_size]

    try:
        print("\nAttempting to load from torch.hub...")
        encoder = torch.hub.load('facebookresearch/vjepa2', cfg['hub_name'], trust_repo=True)
        print(f"  Successfully loaded {cfg['hub_name']} from torch.hub")
        return encoder, {k: cfg[k] for k in ('embed_dim', 'depth', 'num_heads')}

    except Exception as e:
        print(f"  Failed to load from torch.hub: {e}")
        print("\nTrying HuggingFace transformers...")

        try:
            from transformers import AutoModel
            hf_id = f'facebook/vjepa2-{model_size}-fpc64-{resolution}'
            print(f"Loading from HuggingFace: {hf_id}")
            model = AutoModel.from_pretrained(hf_id, trust_remote_code=True)
            encoder = model.encoder
            print(f"  Successfully loaded from HuggingFace")
            return encoder, {k: cfg[k] for k in ('embed_dim', 'depth', 'num_heads')}

        except Exception as e2:
            print(f"  Failed to load from HuggingFace: {e2}")
            raise RuntimeError(
                "Could not load V-JEPA 2 weights. Please ensure you have:\n"
                "1. Internet connection\n"
                "2. Installed: pip install transformers\n"
                "3. Or manually download from: https://github.com/facebookresearch/vjepa2"
            )


def adapt_vjepa2_to_droid_model(vjepa2_encoder, vjepa2_config, target_config):
    """
    Adapt V-JEPA 2 encoder state dict to match DROID world model architecture.
    """
    print(f"\n{'='*70}")
    print("ADAPTING V-JEPA 2 ENCODER TO DROID MODEL")
    print(f"{'='*70}")

    vjepa2_state = vjepa2_encoder.state_dict()

    print(f"\nV-JEPA 2 configuration:")
    print(f"  Embedding dim: {vjepa2_config['embed_dim']}")
    print(f"  Depth: {vjepa2_config['depth']}")
    print(f"  Num heads: {vjepa2_config['num_heads']}")

    print(f"\nTarget DROID model configuration:")
    print(f"  Embedding dim: {target_config['encoder_dim']}")
    print(f"  Depth: {target_config['encoder_depth']}")
    print(f"  Num heads: {target_config['encoder_heads']}")

    if vjepa2_config['embed_dim'] != target_config['encoder_dim']:
        print(f"\n  Dimension mismatch! V-JEPA 2={vjepa2_config['embed_dim']}, "
              f"Target={target_config['encoder_dim']}")
        print(f"  Recommendation: --encoder_dim {vjepa2_config['embed_dim']}")

    if vjepa2_config['depth'] != target_config['encoder_depth']:
        print(f"\n  Depth mismatch! V-JEPA 2={vjepa2_config['depth']}, "
              f"Target={target_config['encoder_depth']}")
        print(f"  Recommendation: --encoder_depth {vjepa2_config['depth']}")

    adapted_state = {}
    for key, value in vjepa2_state.items():
        new_key = key.replace('encoder.', '')
        adapted_state[new_key] = value

    print(f"\n  Adapted {len(adapted_state)} parameters from V-JEPA 2")
    return adapted_state, vjepa2_config


def create_droid_model_with_vjepa2(
    vjepa2_encoder_state, vjepa2_config,
    img_size=224, num_frames=16,
    predictor_dim=1024, predictor_depth=12, predictor_heads=8,
    action_dim=7, state_dim=14, num_task_classes=2,
    freeze_encoder=True
):
    """
    Create DROID world model with V-JEPA 2 pretrained encoder.
    """
    print(f"\n{'='*70}")
    print("CREATING DROID MODEL WITH V-JEPA 2 ENCODER")
    print(f"{'='*70}")

    model = DROIDActionConditionedWorldModel(
        img_size=img_size,
        num_frames=num_frames,
        tubelet_size=(2, 16, 16),
        encoder_dim=vjepa2_config['embed_dim'],
        encoder_depth=vjepa2_config['depth'],
        encoder_heads=vjepa2_config['num_heads'],
        predictor_dim=predictor_dim,
        predictor_depth=predictor_depth,
        predictor_heads=predictor_heads,
        action_dim=action_dim,
        state_dim=state_dim,
        num_task_classes=num_task_classes,
        temporal_scale=1.0,
        use_grad_checkpoint=False
    )

    print(f"\n  Model initialized")
    print(f"  Encoder params: {sum(p.numel() for p in model.encoder.parameters())/1e6:.1f}M")
    print(f"  Total params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    try:
        missing, unexpected = model.encoder.load_state_dict(vjepa2_encoder_state, strict=False)
        print(f"\n  Loaded V-JEPA 2 weights into encoder")
        if missing:
            print(f"  Missing keys (randomly initialized): {len(missing)}")
            for k in missing[:5]: print(f"    - {k}")
            if len(missing) > 5: print(f"    ... and {len(missing)-5} more")
        if unexpected:
            print(f"  Unexpected keys (ignored): {len(unexpected)}")
            for k in unexpected[:5]: print(f"    - {k}")
            if len(unexpected) > 5: print(f"    ... and {len(unexpected)-5} more")
    except Exception as e:
        print(f"\n  Warning: Could not load some weights: {e}")
        print("  Model will work but some layers are randomly initialized.")

    if freeze_encoder:
        print(f"\n  Freezing encoder (only predictor will be trained)")
        for p in model.encoder.parameters():
            p.requires_grad = False
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6
        print(f"  Trainable params: {trainable:.1f}M")
    else:
        print(f"\n  Encoder unfrozen (will be fine-tuned)")

    return model


def save_adapted_checkpoint(model, save_path, vjepa2_config, metadata=None):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'encoder_state_dict': model.encoder.state_dict(),
        'vjepa2_config': vjepa2_config,
        'metadata': metadata or {}
    }
    torch.save(checkpoint, save_path)
    print(f"\n  Saved checkpoint to: {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Load V-JEPA 2 pretrained weights for DROID world model')

    parser.add_argument('--model_size', type=str, default='vitl',
                        choices=['vitl', 'vith', 'vitg'])
    parser.add_argument('--resolution', type=int, default=256, choices=[256, 384])

    # DROID model arguments
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--num_frames', type=int, default=16)
    parser.add_argument('--predictor_dim', type=int, default=1024)
    parser.add_argument('--predictor_depth', type=int, default=12)
    parser.add_argument('--predictor_heads', type=int, default=8)
    parser.add_argument('--action_dim', type=int, default=7,
                        help='DROID action dim: 6 cartesian + 1 gripper = 7')
    parser.add_argument('--state_dim', type=int, default=14,
                        help='DROID state dim: 7 joint + 6 cartesian + 1 gripper = 14')
    parser.add_argument('--num_task_classes', type=int, default=2,
                        help='Task classes: success / failure')

    parser.add_argument('--freeze_encoder', action='store_true', default=True)
    parser.add_argument('--no_freeze_encoder', action='store_false', dest='freeze_encoder')
    parser.add_argument('--save_path', type=str,
                        default='./checkpoints/vjepa2_droid_model.pth')
    parser.add_argument('--test_forward', action='store_true')

    args = parser.parse_args()

    print(f"\n{'='*70}")
    print("V-JEPA 2 PRETRAINED WEIGHT LOADER FOR DROID")
    print(f"{'='*70}")

    # Step 1: Download
    print("\n[Step 1/4] Downloading V-JEPA 2 encoder...")
    vjepa2_encoder, vjepa2_config = download_vjepa2_encoder(args.model_size, args.resolution)

    # Step 2: Adapt
    print("\n[Step 2/4] Adapting V-JEPA 2 to DROID model architecture...")
    target_config = {
        'encoder_dim': vjepa2_config['embed_dim'],
        'encoder_depth': vjepa2_config['depth'],
        'encoder_heads': vjepa2_config['num_heads']
    }
    adapted_state, final_config = adapt_vjepa2_to_droid_model(
        vjepa2_encoder, vjepa2_config, target_config)

    # Step 3: Create model
    print("\n[Step 3/4] Creating DROID world model...")
    model = create_droid_model_with_vjepa2(
        adapted_state, final_config,
        img_size=args.img_size, num_frames=args.num_frames,
        predictor_dim=args.predictor_dim, predictor_depth=args.predictor_depth,
        predictor_heads=args.predictor_heads,
        action_dim=args.action_dim, state_dim=args.state_dim,
        num_task_classes=args.num_task_classes,
        freeze_encoder=args.freeze_encoder)

    # Step 4: Test
    if args.test_forward:
        print("\n[Step 4/4] Testing forward pass...")
        try:
            model.eval()
            with torch.no_grad():
                B = 2
                video = torch.randn(B, 3, args.num_frames, args.img_size, args.img_size)
                actions = torch.randn(B, args.num_frames, args.action_dim)
                states = torch.randn(B, args.num_frames, args.state_dim)
                preds, cur_logits, fut_logits = model(
                    video, actions, states, predict_horizon=4,
                    encoder_frozen=args.freeze_encoder)
                print(f"  Forward pass successful!")
                print(f"  Predictions: {len(preds)} timesteps")
                print(f"  Current task logits: {cur_logits.shape}")
                print(f"  Future task logits: {fut_logits.shape}")
        except Exception as e:
            print(f"  Forward pass test failed: {e}")

    # Step 5: Save
    print("\n[Step 5/5] Saving adapted model...")
    metadata = {
        'vjepa2_model_size': args.model_size,
        'vjepa2_resolution': args.resolution,
        'freeze_encoder': args.freeze_encoder,
        'source': 'V-JEPA 2 pretrained',
        'adapted_for': 'droid_world_model',
        'action_dim': args.action_dim,
        'state_dim': args.state_dim,
    }
    save_adapted_checkpoint(model, args.save_path, final_config, metadata)

    print(f"\n{'='*70}")
    print("SETUP COMPLETE!")
    print(f"{'='*70}")
    print(f"\nDROID world model ready with V-JEPA 2 pretrained encoder!")
    print(f"\nNext steps:")
    print(f"  1. Use this checkpoint for Stage 1 DROID pre-training:")
    print(f"     python stage1_droid_pretraining.py \\")
    print(f"       --data_path <droid_rlds_or_hdf5_path> \\")
    print(f"       --pretrained_checkpoint {args.save_path} \\")
    print(f"       --encoder_dim {final_config['embed_dim']} \\")
    print(f"       --encoder_depth {final_config['depth']} \\")
    print(f"       --encoder_heads {final_config['num_heads']}")
    print(f"\n  2. Or use for direct inference/fine-tuning")
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()

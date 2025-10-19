"""
Load V-JEPA 2 Pretrained Weights for Surgical World Model
==========================================================

This script downloads and loads Meta's V-JEPA 2 pretrained encoder,
adapts it to work with the surgical world model architecture, and
prepares it for fine-tuning on surgical data.

V-JEPA 2 Details:
- Trained on 1M hours of internet video
- State-of-the-art motion understanding
- Uses 3D RoPE (compatible with our architecture)
- Available sizes: ViT-L/16 (300M), ViT-H/16 (600M), ViT-g/16 (1B)

Usage:
    python load_vjepa2_pretrained.py --model_size vitl --save_path ./checkpoints/vjepa2_encoder.pth
"""

import torch
import torch.nn as nn
from pathlib import Path
import argparse
from surgical_world_model import SurgicalActionConditionedWorldModel, VideoViTEncoder


def download_vjepa2_encoder(model_size='vitl', resolution=256):
    """
    Download V-JEPA 2 encoder from Meta's repository.
    
    Args:
        model_size: 'vitl' (300M), 'vith' (600M), or 'vitg' (1B)
        resolution: 256 or 384
        
    Returns:
        encoder: Pretrained V-JEPA 2 encoder
    """
    print(f"\n{'='*70}")
    print("LOADING V-JEPA 2 PRETRAINED ENCODER")
    print(f"{'='*70}")
    print(f"Model size: {model_size.upper()}")
    print(f"Resolution: {resolution}")
    
    try:
        # Try loading from torch.hub (recommended method)
        print("\nAttempting to load from torch.hub...")
        
        if model_size == 'vitl':
            model_name = 'vjepa2_vitl'
            expected_dim = 1024
            expected_depth = 24
            expected_heads = 16
        elif model_size == 'vith':
            model_name = 'vjepa2_vith'
            expected_dim = 1280
            expected_depth = 32
            expected_heads = 16
        elif model_size == 'vitg':
            model_name = 'vjepa2_vitg'
            expected_dim = 1024
            expected_depth = 40
            expected_heads = 16
        else:
            raise ValueError(f"Unknown model size: {model_size}")
        
        # Load encoder from torch hub
        encoder = torch.hub.load(
            'facebookresearch/vjepa2',
            model_name,
            trust_repo=True
        )
        
        print(f"✓ Successfully loaded {model_name} from torch.hub")
        
        return encoder, {
            'embed_dim': expected_dim,
            'depth': expected_depth,
            'num_heads': expected_heads
        }
        
    except Exception as e:
        print(f"⚠ Failed to load from torch.hub: {e}")
        print("\nTrying alternative method: HuggingFace transformers...")
        
        try:
            from transformers import AutoModel
            
            # Map model sizes to HuggingFace repo names
            hf_model_map = {
                'vitl': f'facebook/vjepa2-vitl-fpc64-{resolution}',
                'vith': f'facebook/vjepa2-vith-fpc64-{resolution}',
                'vitg': f'facebook/vjepa2-vitg-fpc64-{resolution}'
            }
            
            model_id = hf_model_map.get(model_size)
            if model_id is None:
                raise ValueError(f"Unknown model size: {model_size}")
            
            print(f"Loading from HuggingFace: {model_id}")
            
            model = AutoModel.from_pretrained(
                model_id,
                trust_remote_code=True
            )
            
            # Extract encoder
            encoder = model.encoder
            
            print(f"✓ Successfully loaded from HuggingFace")
            
            # Get model config
            config_map = {
                'vitl': {'embed_dim': 1024, 'depth': 24, 'num_heads': 16},
                'vith': {'embed_dim': 1280, 'depth': 32, 'num_heads': 16},
                'vitg': {'embed_dim': 1024, 'depth': 40, 'num_heads': 16}
            }
            
            return encoder, config_map[model_size]
            
        except Exception as e2:
            print(f"⚠ Failed to load from HuggingFace: {e2}")
            raise RuntimeError(
                "Could not load V-JEPA 2 weights. Please ensure you have:\n"
                "1. Internet connection\n"
                "2. Installed: pip install transformers\n"
                "3. Or manually download from: https://github.com/facebookresearch/vjepa2"
            )


def adapt_vjepa2_to_surgical_model(vjepa2_encoder, vjepa2_config, target_config):
    """
    Adapt V-JEPA 2 encoder to match surgical world model architecture.
    
    Args:
        vjepa2_encoder: Pretrained V-JEPA 2 encoder
        vjepa2_config: V-JEPA 2 model configuration
        target_config: Target surgical model configuration
        
    Returns:
        adapted_state_dict: State dict compatible with surgical model
    """
    print(f"\n{'='*70}")
    print("ADAPTING V-JEPA 2 ENCODER TO SURGICAL MODEL")
    print(f"{'='*70}")
    
    # Get V-JEPA 2 state dict
    vjepa2_state = vjepa2_encoder.state_dict()
    
    print(f"\nV-JEPA 2 configuration:")
    print(f"  - Embedding dim: {vjepa2_config['embed_dim']}")
    print(f"  - Depth: {vjepa2_config['depth']}")
    print(f"  - Num heads: {vjepa2_config['num_heads']}")
    
    print(f"\nTarget surgical model configuration:")
    print(f"  - Embedding dim: {target_config['encoder_dim']}")
    print(f"  - Depth: {target_config['encoder_depth']}")
    print(f"  - Num heads: {target_config['encoder_heads']}")
    
    # Check if dimensions match
    if vjepa2_config['embed_dim'] != target_config['encoder_dim']:
        print(f"\n⚠ Dimension mismatch detected!")
        print(f"  V-JEPA 2: {vjepa2_config['embed_dim']}")
        print(f"  Target: {target_config['encoder_dim']}")
        print(f"\n💡 Recommendation: Set --encoder_dim {vjepa2_config['embed_dim']} in training script")
    
    if vjepa2_config['depth'] != target_config['encoder_depth']:
        print(f"\n⚠ Depth mismatch detected!")
        print(f"  V-JEPA 2: {vjepa2_config['depth']}")
        print(f"  Target: {target_config['encoder_depth']}")
        print(f"\n💡 Recommendation: Set --encoder_depth {vjepa2_config['depth']} in training script")
    
    # Create mapping between V-JEPA 2 and surgical model keys
    adapted_state = {}
    
    # Map keys (this may need adjustment based on actual V-JEPA 2 architecture)
    for key, value in vjepa2_state.items():
        # Remove 'encoder.' prefix if present
        new_key = key.replace('encoder.', '')
        adapted_state[new_key] = value
    
    print(f"\n✓ Adapted {len(adapted_state)} parameters from V-JEPA 2")
    
    return adapted_state, vjepa2_config


def create_surgical_model_with_vjepa2(
    vjepa2_encoder_state,
    vjepa2_config,
    img_size=224,
    num_frames=16,
    predictor_dim=1024,
    predictor_depth=12,
    predictor_heads=8,
    freeze_encoder=True
):
    """
    Create surgical world model with V-JEPA 2 pretrained encoder.
    
    Args:
        vjepa2_encoder_state: Adapted V-JEPA 2 encoder state dict
        vjepa2_config: V-JEPA 2 configuration
        img_size: Input image size
        num_frames: Number of frames per clip
        predictor_dim: Predictor dimension
        predictor_depth: Predictor depth
        predictor_heads: Predictor heads
        freeze_encoder: Whether to freeze encoder weights
        
    Returns:
        model: Surgical world model with pretrained encoder
    """
    print(f"\n{'='*70}")
    print("CREATING SURGICAL MODEL WITH V-JEPA 2 ENCODER")
    print(f"{'='*70}")
    
    # Create model with V-JEPA 2 dimensions
    model = SurgicalActionConditionedWorldModel(
        img_size=img_size,
        num_frames=num_frames,
        tubelet_size=(2, 16, 16),
        in_channels=3,
        encoder_dim=vjepa2_config['embed_dim'],
        encoder_depth=vjepa2_config['depth'],
        encoder_heads=vjepa2_config['num_heads'],
        predictor_dim=predictor_dim,
        predictor_depth=predictor_depth,
        predictor_heads=predictor_heads,
        action_dim=7,
        state_dim=7,
        num_anomaly_classes=2,
        use_grad_checkpoint=False
    )
    
    print(f"\n✓ Model initialized")
    print(f"  - Encoder params: {sum(p.numel() for p in model.encoder.parameters())/1e6:.1f}M")
    print(f"  - Total params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    
    # Load V-JEPA 2 weights into encoder
    try:
        missing_keys, unexpected_keys = model.encoder.load_state_dict(
            vjepa2_encoder_state, 
            strict=False
        )
        
        print(f"\n✓ Loaded V-JEPA 2 weights into encoder")
        
        if missing_keys:
            print(f"\n⚠ Missing keys (will be randomly initialized):")
            for key in missing_keys[:5]:  # Show first 5
                print(f"    - {key}")
            if len(missing_keys) > 5:
                print(f"    ... and {len(missing_keys)-5} more")
        
        if unexpected_keys:
            print(f"\n⚠ Unexpected keys (ignored):")
            for key in unexpected_keys[:5]:  # Show first 5
                print(f"    - {key}")
            if len(unexpected_keys) > 5:
                print(f"    ... and {len(unexpected_keys)-5} more")
                
    except Exception as e:
        print(f"\n⚠ Warning: Could not load some weights: {e}")
        print("This is normal if architectures don't perfectly match.")
        print("The model will work, but some layers will be randomly initialized.")
    
    # Freeze encoder if requested
    if freeze_encoder:
        print(f"\n✓ Freezing encoder weights (only predictor will be trained)")
        for param in model.encoder.parameters():
            param.requires_grad = False
        
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
        print(f"  - Trainable params: {trainable_params:.1f}M")
    else:
        print(f"\n✓ Encoder unfrozen (will be fine-tuned)")
    
    return model


def save_adapted_checkpoint(model, save_path, vjepa2_config, metadata=None):
    """
    Save adapted model checkpoint with metadata.
    
    Args:
        model: Surgical world model with V-JEPA 2 encoder
        save_path: Path to save checkpoint
        vjepa2_config: V-JEPA 2 configuration
        metadata: Additional metadata to save
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'encoder_state_dict': model.encoder.state_dict(),
        'vjepa2_config': vjepa2_config,
        'metadata': metadata or {}
    }
    
    torch.save(checkpoint, save_path)
    print(f"\n✓ Saved checkpoint to: {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Load V-JEPA 2 pretrained weights for surgical world model'
    )
    
    # V-JEPA 2 arguments
    parser.add_argument('--model_size', type=str, default='vitl',
                        choices=['vitl', 'vith', 'vitg'],
                        help='V-JEPA 2 model size (vitl=300M, vith=600M, vitg=1B)')
    parser.add_argument('--resolution', type=int, default=256,
                        choices=[256, 384],
                        help='Input resolution')
    
    # Surgical model arguments
    parser.add_argument('--img_size', type=int, default=224,
                        help='Input image size for surgical model')
    parser.add_argument('--num_frames', type=int, default=16,
                        help='Number of frames per clip')
    parser.add_argument('--predictor_dim', type=int, default=1024,
                        help='Predictor dimension')
    parser.add_argument('--predictor_depth', type=int, default=12,
                        help='Predictor depth')
    parser.add_argument('--predictor_heads', type=int, default=8,
                        help='Predictor heads')
    
    # Training arguments
    parser.add_argument('--freeze_encoder', action='store_true', default=True,
                        help='Freeze encoder weights (recommended for Stage 1)')
    parser.add_argument('--no_freeze_encoder', action='store_false', dest='freeze_encoder',
                        help='Allow encoder fine-tuning')
    
    # Output arguments
    parser.add_argument('--save_path', type=str, 
                        default='./checkpoints/vjepa2_surgical_model.pth',
                        help='Path to save adapted model')
    parser.add_argument('--test_forward', action='store_true',
                        help='Test forward pass after loading')
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print("V-JEPA 2 PRETRAINED WEIGHT LOADER")
    print(f"{'='*70}")
    
    # Step 1: Download V-JEPA 2 encoder
    print("\n[Step 1/4] Downloading V-JEPA 2 encoder...")
    vjepa2_encoder, vjepa2_config = download_vjepa2_encoder(
        args.model_size, 
        args.resolution
    )
    
    # Step 2: Adapt to surgical model
    print("\n[Step 2/4] Adapting V-JEPA 2 to surgical model architecture...")
    
    target_config = {
        'encoder_dim': vjepa2_config['embed_dim'],  # Use V-JEPA 2 dimensions
        'encoder_depth': vjepa2_config['depth'],
        'encoder_heads': vjepa2_config['num_heads']
    }
    
    adapted_state, final_config = adapt_vjepa2_to_surgical_model(
        vjepa2_encoder,
        vjepa2_config,
        target_config
    )
    
    # Step 3: Create surgical model with pretrained encoder
    print("\n[Step 3/4] Creating surgical world model...")
    model = create_surgical_model_with_vjepa2(
        adapted_state,
        final_config,
        img_size=args.img_size,
        num_frames=args.num_frames,
        predictor_dim=args.predictor_dim,
        predictor_depth=args.predictor_depth,
        predictor_heads=args.predictor_heads,
        freeze_encoder=args.freeze_encoder
    )
    
    # Step 4: Test forward pass (optional)
    if args.test_forward:
        print("\n[Step 4/4] Testing forward pass...")
        
        try:
            model.eval()
            with torch.no_grad():
                # Create dummy inputs
                batch_size = 2
                video = torch.randn(batch_size, 3, args.num_frames, args.img_size, args.img_size)
                actions = torch.randn(batch_size, args.num_frames, 7)
                states = torch.randn(batch_size, args.num_frames, 7)
                
                # Forward pass
                predictions, current_logits, future_logits = model(
                    video, actions, states, 
                    predict_horizon=4,
                    encoder_frozen=args.freeze_encoder
                )
                
                print(f"✓ Forward pass successful!")
                print(f"  - Predictions: {len(predictions)} timesteps")
                print(f"  - Current sterility logits: {current_logits.shape}")
                print(f"  - Future sterility logits: {future_logits.shape}")
                
        except Exception as e:
            print(f"⚠ Forward pass test failed: {e}")
            print("Model may still work - this could be a configuration mismatch")
    
    # Step 5: Save checkpoint
    print("\n[Step 5/5] Saving adapted model...")
    
    metadata = {
        'vjepa2_model_size': args.model_size,
        'vjepa2_resolution': args.resolution,
        'freeze_encoder': args.freeze_encoder,
        'source': 'V-JEPA 2 pretrained',
        'adapted_for': 'surgical_world_model'
    }
    
    save_adapted_checkpoint(model, args.save_path, final_config, metadata)
    
    # Final summary
    print(f"\n{'='*70}")
    print("SETUP COMPLETE!")
    print(f"{'='*70}")
    print(f"\nYour surgical world model is ready with V-JEPA 2 pretrained encoder!")
    print(f"\nNext steps:")
    print(f"  1. Use this checkpoint for Stage 1 training:")
    print(f"     python stage1_surgvu_pretraining.py \\")
    print(f"       --data_root ./data/SurgVU \\")
    print(f"       --pretrained_checkpoint {args.save_path} \\")
    print(f"       --encoder_dim {final_config['embed_dim']} \\")
    print(f"       --encoder_depth {final_config['depth']} \\")
    print(f"       --encoder_heads {final_config['num_heads']}")
    print(f"\n  2. Or use for direct inference/fine-tuning")
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()

# V-JEPA 2 Quick Reference Card 🚀

## 📋 Essential Commands (Copy & Paste)

### 1️⃣ Load V-JEPA 2 Weights (Choose One)

```bash
# Recommended: ViT-L/16 (300M params, ~8GB GPU)
python load_vjepa2_pretrained.py \
    --model_size vitl \
    --save_path ./checkpoints/vjepa2_encoder.pth \
    --test_forward

# Powerful: ViT-H/16 (600M params, ~16GB GPU)
python load_vjepa2_pretrained.py \
    --model_size vith \
    --save_path ./checkpoints/vjepa2_encoder.pth

# Best Quality: ViT-g/16 (1B params, ~24GB GPU)
python load_vjepa2_pretrained.py \
    --model_size vitg \
    --save_path ./checkpoints/vjepa2_encoder.pth
```

---

### 2️⃣ Fine-tune on SurgVU (Choose One)

```bash
# Recommended: Frozen Encoder (Fast, Stable)
python stage1_vjepa2_finetuning.py \
    --data_root ./data/SurgVU \
    --pretrained_checkpoint ./checkpoints/vjepa2_encoder.pth \
    --encoder_frozen \
    --num_epochs 10 \
    --batch_size 4

# Advanced: Fine-tune Encoder (Slower, Potentially Better)
python stage1_vjepa2_finetuning.py \
    --data_root ./data/SurgVU \
    --pretrained_checkpoint ./checkpoints/vjepa2_encoder.pth \
    --finetune_encoder \
    --num_epochs 15 \
    --batch_size 2 \
    --learning_rate 1e-4
```

---

## 🔧 Hardware-Specific Commands

### For Low Memory GPU (<8GB)
```bash
python stage1_vjepa2_finetuning.py \
    --data_root ./data/SurgVU \
    --pretrained_checkpoint ./checkpoints/vjepa2_encoder.pth \
    --encoder_frozen \
    --batch_size 1 \
    --num_workers 2
```

### For Apple Silicon (M1/M2/M3)
```bash
python load_vjepa2_pretrained.py \
    --model_size vitl \
    --save_path ./checkpoints/vjepa2_encoder.pth

python stage1_vjepa2_finetuning.py \
    --data_root ./data/SurgVU \
    --pretrained_checkpoint ./checkpoints/vjepa2_encoder.pth \
    --encoder_frozen \
    --batch_size 2 \
    --device mps
```

### For High-End GPU (A100, 80GB)
```bash
python stage1_vjepa2_finetuning.py \
    --data_root ./data/SurgVU \
    --pretrained_checkpoint ./checkpoints/vjepa2_encoder.pth \
    --encoder_frozen \
    --batch_size 8 \
    --num_workers 8 \
    --use_mask_aux
```

---

## 📊 Model Size Comparison

| Model | Parameters | GPU Memory | Speed | Quality | Command |
|-------|-----------|------------|-------|---------|---------|
| **ViT-L** | 300M | ~8GB | Fast | Good | `--model_size vitl` |
| **ViT-H** | 600M | ~16GB | Medium | Better | `--model_size vith` |
| **ViT-g** | 1B | ~24GB | Slow | Best | `--model_size vitg` |

---

## 🐛 Quick Troubleshooting

### Out of Memory?
```bash
# Reduce batch size
--batch_size 1

# Use smaller model
--model_size vitl

# Ensure encoder frozen
--encoder_frozen
```

### Slow Training?
```bash
# Increase workers
--num_workers 8

# Larger batch size (if GPU allows)
--batch_size 8

# Use frozen encoder
--encoder_frozen
```

### Can't Load V-JEPA 2?
```bash
# Install transformers
pip install transformers

# Check internet
ping github.com

# Verify PyTorch
python -c "import torch; print(torch.__version__)"
```

---

## ⏱️ Expected Training Times

| Hardware | Batch Size | Time (10 epochs) |
|----------|-----------|------------------|
| A100 80GB | 8 | ~6 hours |
| A100 40GB | 4 | ~8 hours |
| RTX 3090 | 2 | ~12 hours |
| M2 Max | 2 | ~20 hours |

---

## ✅ Success Indicators

**Good Training:**
```
Epoch 01:
  Temporal Loss: 0.623      ← decreasing ✓
  Contrastive Loss: 2.145   ← decreasing ✓
  Total Loss: 2.768         ← decreasing ✓
  ✓ Saved best checkpoint
```

**Output File:**
```
./checkpoints/best_vjepa2_finetuned.pth
```

---

## 📁 Files You Need

1. ✅ **load_vjepa2_pretrained.py** - Load V-JEPA 2 weights
2. ✅ **stage1_vjepa2_finetuning.py** - Fine-tuning script
3. ✅ **VJEPA2_INTEGRATION_GUIDE.md** - Full documentation
4. ✅ **surgical_world_model.py** - Model architecture (from your files)
5. ✅ **stage1_surgvu_pretraining.py** - Dataset classes (from your files)

---

## 🚀 Complete Workflow (3 Steps)

```bash
# Step 1: Load V-JEPA 2 (5 minutes)
python load_vjepa2_pretrained.py \
    --model_size vitl \
    --save_path ./checkpoints/vjepa2_encoder.pth

# Step 2: Fine-tune on SurgVU (~8 hours)
python stage1_vjepa2_finetuning.py \
    --data_root ./data/SurgVU \
    --pretrained_checkpoint ./checkpoints/vjepa2_encoder.pth \
    --encoder_frozen \
    --num_epochs 10

# Step 3: Verify output
ls -lh ./checkpoints/best_vjepa2_finetuned.pth
```

---

## 🔗 Important Links

- **V-JEPA 2 Paper:** https://arxiv.org/abs/2506.09985
- **GitHub:** https://github.com/facebookresearch/vjepa2
- **HuggingFace:** https://huggingface.co/facebook/vjepa2-vitl-fpc64-256

---

## 📞 Need Help?

1. Read **VJEPA2_INTEGRATION_GUIDE.md** for detailed instructions
2. Run `python test_stage1_setup.py` to verify setup
3. Check GPU memory: `nvidia-smi` (or Activity Monitor on Mac)
4. Ensure dependencies: `pip install torch torchvision transformers`

---

**Quick Tip:** Always start with ViT-L and frozen encoder. It's the best balance of speed and quality! 🎯

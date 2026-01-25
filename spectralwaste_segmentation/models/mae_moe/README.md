# MAE-MoE: Self-Supervised Pretrained MoE for RGB+HSI Segmentation

## Overview

MAE-MoE combines **Masked Autoencoder (MAE)** pretraining with **Mixture of Experts (MoE)** routing for RGB+HSI waste segmentation. This model leverages:

1. **Self-supervised pretraining** via MAE on unlabeled data
2. **Dual-branch HSI encoding** for spatial and channel dimensions
3. **Adaptive multimodal fusion** via sparse MoE routing
4. **Expert specialization** for different modality combinations

## Architecture

```
RGB Image (B, 3, H, W)          HSI Image (B, C_hsi, H, W)
        ↓                                    ↓
   ViT Encoder                    Dual-Branch MAE Encoder
        ↓                           ↓                ↓
  RGB Features              HSI Spatial      HSI Channel
   (B, N, D)                 Features          Features
        ↓                    (B, N, D)         (B, N, D)
        └──────────────────────┴──────────────────┘
                              ↓
                    MoE Fusion (Per-Modality)
                              ↓
                      Fused Features (B, N, D)
                              ↓
                      Segmentation Head
                              ↓
                    Output (B, num_classes, H, W)
```

## Key Components

### 1. RGB Encoder
- Standard Vision Transformer (ViT)
- Patch-based encoding with positional embeddings
- Can be pretrained with MAE on RGB images

### 2. HSI Encoder (Dual-Branch)
- **Spatial Branch**: Processes spatial patterns across all bands
- **Channel Branch**: Processes spectral signatures across channels
- Based on SS-MAE architecture

### 3. MoE Fusion
Three router types available:
- **`joint`**: Single router for concatenated modalities
- **`permod`**: Per-modality routers with shared expert pool
- **`disjoint`**: Separate routers and experts per modality

Gating functions:
- **`softmax`**: Standard softmax gating
- **`laplace`**: Laplace distance-based gating
- **`gaussian`**: Gaussian distance-based gating

### 4. Segmentation Head
- Upsampling from patch-level to pixel-level
- Conv layers for final class prediction

## Usage

### Basic Usage

```python
from spectralwaste_segmentation.models.mae_moe import MAEMoE

# Create model
model = MAEMoE(
    img_size=224,
    patch_size=16,
    rgb_channels=3,
    hsi_channels=150,
    num_classes=10,
    embed_dim=768,
    depth=12,
    num_heads=12,
    num_experts=8,
    top_k=4,
    router_type='permod',  # 'joint', 'permod', or 'disjoint'
    gating='softmax',
    noisy_gating=True,
)

# Forward pass
rgb = torch.randn(2, 3, 224, 224)
hsi = torch.randn(2, 150, 224, 224)

# Training
output, aux_loss = model(rgb, hsi, train=True)
# output: [2, 10, 224, 224]
# aux_loss: scalar (load balancing loss)

# Inference
output = model(rgb, hsi, train=False)
# output: [2, 10, 224, 224]
```

### Training with Load Balancing Loss

```python
# Main segmentation loss
seg_loss = criterion(output, target)

# MoE load balancing loss
aux_loss = model.aux_loss  # or from forward return

# Total loss
total_loss = seg_loss + 1e-2 * aux_loss  # loss_coef=1e-2
```

## Model Configurations

### Small Model (Fast)
```python
model = MAEMoE(
    embed_dim=384,
    depth=6,
    num_heads=6,
    num_experts=4,
    top_k=2,
)
```

### Base Model (Balanced)
```python
model = MAEMoE(
    embed_dim=768,
    depth=12,
    num_heads=12,
    num_experts=8,
    top_k=4,
)
```

### Large Model (Accurate)
```python
model = MAEMoE(
    embed_dim=1024,
    depth=24,
    num_heads=16,
    num_experts=16,
    top_k=4,
)
```

## Pretraining (Optional)

MAE-MoE supports self-supervised pretraining on unlabeled data:

1. **Pretrain RGB Encoder** with MAE on RGB images
2. **Pretrain HSI Encoder** with SS-MAE on HSI data
3. **Finetune** the full model on labeled segmentation data

## Files

- `model.py`: Main MAE-MoE model
- `vit_encoder.py`: Vision Transformer encoder for RGB
- `mae_encoder.py`: Dual-branch MAE encoder for HSI
- `sparse_moe.py`: Sparse Mixture of Experts implementation
- `config.py`: MoE configuration
- `patch_embed.py`: Patch embedding layers
- `vit_layers.py`: ViT building blocks
- `activations.py`: Activation functions

## References

- **FuseMoE**: Original MoE fusion implementation
- **SS-MAE**: Self-supervised MAE for hyperspectral images
- **ViT**: Vision Transformer (Dosovitskiy et al., 2020)
- **MAE**: Masked Autoencoders (He et al., 2021)


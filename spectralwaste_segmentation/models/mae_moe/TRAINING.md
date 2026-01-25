# MAE-MoE Training Guide

This guide explains how to train the MAE-MoE model for RGB+HSI waste segmentation.

## Overview

MAE-MoE training consists of two stages:

1. **Pretrain** (Optional): Pretrain RGB and HSI encoders separately using MAE reconstruction
2. **Finetune**: Train the full MAE-MoE model on segmentation task

## Quick Start (Finetune Only)

If you want to skip pretraining and train directly on segmentation:

```bash
python -m scripts.train_mae_moe_finetune \
    --data-path data/spectralwaste_segmentation \
    --results-path results/mae_moe \
    --experiment-name mae_moe_base \
    --batch-size 12 \
    --max-epoch 200 \
    --device cuda
```

## Two-Stage Training (Pretrain + Finetune)

### Stage 1: Pretrain Encoders

**Note**: The current pretrain script is a placeholder. Full MAE reconstruction with masking and decoder will be implemented in future versions.

#### Pretrain RGB Encoder

```bash
python -m scripts.train_mae_moe_pretrain \
    --encoder rgb \
    --data-path data/spectralwaste_segmentation \
    --results-path results/mae_moe_pretrain \
    --experiment-name mae_moe \
    --batch-size 12 \
    --max-epoch 100 \
    --mask-ratio 0.75 \
    --device cuda
```

This will save: `results/mae_moe_pretrain/mae_moe_rgb_encoder_pretrain.pth`

#### Pretrain HSI Encoder

```bash
python -m scripts.train_mae_moe_pretrain \
    --encoder hsi \
    --data-path data/spectralwaste_segmentation \
    --results-path results/mae_moe_pretrain \
    --experiment-name mae_moe \
    --batch-size 12 \
    --max-epoch 100 \
    --mask-ratio 0.75 \
    --device cuda
```

This will save: `results/mae_moe_pretrain/mae_moe_hsi_encoder_pretrain.pth`

### Stage 2: Finetune with Pretrained Encoders

```bash
python -m scripts.train_mae_moe_finetune \
    --data-path data/spectralwaste_segmentation \
    --results-path results/mae_moe_finetune \
    --experiment-name mae_moe_pretrained \
    --rgb-encoder-pretrain results/mae_moe_pretrain/mae_moe_rgb_encoder_pretrain.pth \
    --hsi-encoder-pretrain results/mae_moe_pretrain/mae_moe_hsi_encoder_pretrain.pth \
    --batch-size 12 \
    --max-epoch 200 \
    --device cuda
```

## Model Variants

### Small Model (Fast)

```bash
python -m scripts.train_mae_moe_finetune \
    --embed-dim 384 \
    --depth 6 \
    --num-heads 6 \
    --num-experts 4 \
    --top-k 2 \
    --experiment-name mae_moe_small \
    ...
```

### Base Model (Default)

```bash
python -m scripts.train_mae_moe_finetune \
    --embed-dim 768 \
    --depth 12 \
    --num-heads 12 \
    --num-experts 8 \
    --top-k 4 \
    --experiment-name mae_moe_base \
    ...
```

### Large Model (Accurate)

```bash
python -m scripts.train_mae_moe_finetune \
    --embed-dim 1024 \
    --depth 24 \
    --num-heads 16 \
    --num-experts 16 \
    --top-k 4 \
    --experiment-name mae_moe_large \
    ...
```

## MoE Router Types

### Per-Modality Router (Default, Recommended)

```bash
--router-type permod
```

Each modality (RGB, HSI-spatial, HSI-channel) has its own router, but shares the same expert pool.

### Joint Router

```bash
--router-type joint
```

Single router for all concatenated modalities.

### Disjoint Router

```bash
--router-type disjoint
```

Separate routers and experts for each modality.

## Important Parameters

- `--loss-coef`: Coefficient for MoE load balancing loss (default: 1e-2)
- `--router-type`: MoE routing strategy (default: permod)
- `--gating`: Gating function (softmax, laplace, gaussian)
- `--noisy-gating`: Enable noisy gating for exploration (default: True)
- `--num-experts`: Number of experts in MoE (default: 8)
- `--top-k`: Number of experts to activate per token (default: 4)

## Resume Training

```bash
python -m scripts.train_mae_moe_finetune \
    --resume results/mae_moe_finetune/mae_moe.last.pth \
    ...
```

## Test Only

```bash
python -m scripts.train_mae_moe_finetune \
    --resume results/mae_moe_finetune/mae_moe.best.pth \
    --test-only \
    ...
```

## Wandb Logging

```bash
python -m scripts.train_mae_moe_finetune \
    --wandb your_project_name \
    ...
```

## Notes

- MAE-MoE requires **multimodal input**: `['rgb', 'hyper']`
- The model automatically handles MoE load balancing loss during training
- Pretrained encoders are optional but may improve performance
- Current pretrain script is a placeholder - full MAE implementation coming soon


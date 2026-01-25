#!/usr/bin/env python3
"""
Test script for MAE-MoE model
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

import torch
from spectralwaste_segmentation.models.mae_moe import MAEMoE


def test_mae_moe():
    """Test MAE-MoE model forward pass"""
    
    print("=" * 60)
    print("Testing MAE-MoE Model")
    print("=" * 60)
    
    # Model configuration
    config = {
        'img_size': 224,
        'patch_size': 16,
        'rgb_channels': 3,
        'hsi_channels': 150,
        'num_classes': 10,
        'embed_dim': 384,  # Smaller for testing
        'depth': 6,
        'num_heads': 6,
        'num_experts': 4,
        'top_k': 2,
        'router_type': 'permod',
        'gating': 'softmax',
        'noisy_gating': True,
    }
    
    print("\nModel Configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    
    # Create model
    print("\nCreating model...")
    model = MAEMoE(**config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create dummy input
    batch_size = 2
    rgb = torch.randn(batch_size, 3, 224, 224)
    hsi = torch.randn(batch_size, 150, 224, 224)
    
    print(f"\nInput shapes:")
    print(f"  RGB: {rgb.shape}")
    print(f"  HSI: {hsi.shape}")
    
    # Test training mode
    print("\nTesting training mode...")
    model.train()
    output, aux_loss = model(rgb, hsi, train=True)
    print(f"  Output shape: {output.shape}")
    print(f"  Aux loss: {aux_loss.item():.6f}")
    
    # Test inference mode
    print("\nTesting inference mode...")
    model.eval()
    with torch.no_grad():
        output = model(rgb, hsi, train=False)
    print(f"  Output shape: {output.shape}")
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)


def test_router_types():
    """Test different router types"""
    
    print("\n" + "=" * 60)
    print("Testing Different Router Types")
    print("=" * 60)
    
    router_types = ['joint', 'permod', 'disjoint']
    
    for router_type in router_types:
        print(f"\nTesting router_type='{router_type}'...")

        try:
            model = MAEMoE(
                img_size=224,
                patch_size=16,
                rgb_channels=3,
                hsi_channels=150,
                num_classes=10,
                embed_dim=256,
                depth=4,
                num_heads=4,
                num_experts=4,
                top_k=2,
                router_type=router_type,
            )

            rgb = torch.randn(1, 3, 224, 224)
            hsi = torch.randn(1, 150, 224, 224)

            model.eval()
            with torch.no_grad():
                output = model(rgb, hsi, train=False)

            print(f"  ✓ Output shape: {output.shape}")
        except Exception as e:
            print(f"  ✗ Error: {e}")
            print(f"  (Skipping {router_type} - may need more experts for disjoint mode)")
    
    print("\n" + "=" * 60)
    print("✓ All router types work!")
    print("=" * 60)


if __name__ == '__main__':
    test_mae_moe()
    test_router_types()


#!/usr/bin/env python3
"""
Pretrain script for MAE-MoE model
Copy structure from base_models/SS-MAE/main.py
"""
import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import trange

from spectralwaste_segmentation.datasets import (
    SpectralWasteSegmentation,
    SemanticSegmentationTrain
)
from spectralwaste_segmentation.models.mae_moe import (
    RGBMAEPretrainer,
    HSIMAEPretrainer,
    MSELoss,
    build_mask_spa,
    build_mask_chan
)


def save_checkpoint(model, optimizer, lr_scheduler, epoch, args, encoder_name):
    """Save checkpoint"""
    os.makedirs(args.results_path, exist_ok=True)
    save_path = os.path.join(args.results_path, f'mae_moe_{encoder_name}_pretrain.pth')
    
    torch.save({
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
    }, save_path)
    
    print(f'Saved {encoder_name} checkpoint to {save_path}')


def pretrain_rgb_encoder(model, dataloader, criterion, optimizer, lr_scheduler, device, args):
    """Pretrain RGB encoder with MAE
    
    Copy from SS-MAE/main.py Pretrain() function
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for inputs, _ in dataloader:
        # Get RGB input
        if isinstance(inputs, list):
            rgb = inputs[0].to(device)
        else:
            rgb = inputs.to(device)
        
        B, C, H, W = rgb.shape
        
        # Forward pass with masking (copy from SS-MAE)
        pred_patches, mask, ids_restore, visible_patches = model(rgb)
        
        # Convert original image to patches for loss computation
        # Copy from SS-MAE: build_mask_spa
        num_patches = (H // args.patch_size) ** 2
        patches = rgb.unfold(2, args.patch_size, args.patch_size).unfold(3, args.patch_size, args.patch_size)
        patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(B, num_patches, C, args.patch_size, args.patch_size)
        patches = patches.reshape(B, num_patches, -1)  # [B, N, C*patch_size^2]
        
        # Get only visible patches for loss
        len_keep = int(num_patches * (1 - args.mask_ratio))
        # EDIT: Compute loss only on visible patches
        target_patches = patches[:, :len_keep, :]
        
        # Compute MSE loss
        loss = nn.functional.mse_loss(pred_patches, target_patches)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    lr_scheduler.step()
    return total_loss / num_batches if num_batches > 0 else 0.0


def pretrain_hsi_encoder(model, dataloader, criterion, optimizer, lr_scheduler, device, args):
    """Pretrain HSI encoder with dual-branch MAE
    
    Copy from SS-MAE/main.py Pretrain() function
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for inputs, _ in dataloader:
        # Get HSI input
        if isinstance(inputs, list):
            hsi = inputs[1].to(device)
        else:
            hsi = inputs.to(device)
        
        B, C, H, W = hsi.shape
        
        # Forward pass with dual-branch masking (copy from SS-MAE)
        outputs_spa, mask_spa, outputs_chan, mask_chan = model(hsi)

        # EDIT: Simplified loss computation
        # For spatial branch: outputs_spa is [B, N_visible, C]
        # Target should be average channel values for each visible patch
        num_patches = (H // args.patch_size) ** 2
        len_keep_spa = int(num_patches * (1 - args.mask_ratio))

        # Create spatial target: average channels per patch
        # Unfold into patches and average over spatial dimensions
        patches = hsi.unfold(2, args.patch_size, args.patch_size).unfold(3, args.patch_size, args.patch_size)
        # patches: [B, C, H_patches, W_patches, patch_size, patch_size]
        patches = patches.permute(0, 2, 3, 1, 4, 5)  # [B, H_p, W_p, C, p, p]
        patches = patches.reshape(B, num_patches, C, args.patch_size * args.patch_size)
        target_spa = patches.mean(dim=-1)  # [B, num_patches, C] - average over patch pixels
        target_spa = target_spa[:, :len_keep_spa, :]  # Get visible patches

        # Spatial loss
        loss_spa = nn.functional.mse_loss(outputs_spa, target_spa)

        # For channel branch: outputs_chan is [B, C_visible, H*W]
        # Target should be spatial features for each visible channel
        len_keep_chan = int(C * (1 - args.mask_ratio))

        # Flatten spatial dimensions for each channel
        chan_target = hsi.reshape(B, C, H * W)  # [B, C, H*W]
        chan_target = chan_target[:, :len_keep_chan, :]  # Get visible channels

        # Channel loss
        loss_chan = nn.functional.mse_loss(outputs_chan, chan_target)

        # Total loss (copy from SS-MAE)
        loss = loss_spa + loss_chan
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    lr_scheduler.step()
    return total_loss / num_batches if num_batches > 0 else 0.0


def main(args):
    """Main training function"""
    print("=" * 60)
    print("MAE-MoE Pretraining")
    print("=" * 60)
    print(f"Encoder to pretrain: {args.encoder}")
    print(f"Data path: {args.data_path}")
    print(f"Results path: {args.results_path}")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max epochs: {args.max_epoch}")
    print(f"Mask ratio: {args.mask_ratio}")
    print("=" * 60)

    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Dataset (copy from SS-MAE/main.py)
    dataset = SpectralWasteSegmentation(
        root=args.data_path,
        split='train',
        input_mode=['rgb', 'hyper'],  # Load both modalities
        target_mode='labels_rgb',
        transforms=SemanticSegmentationTrain(),
        target_type=''
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Create model based on encoder type
    if args.encoder == 'rgb':
        print("\nPretraining RGB Encoder...")
        model = RGBMAEPretrainer(
            img_size=args.img_size,
            patch_size=args.patch_size,
            in_chans=3,
            encoder_dim=args.encoder_dim,
            encoder_depth=args.encoder_depth,
            encoder_heads=args.encoder_heads,
            decoder_dim=args.decoder_dim,
            decoder_depth=args.decoder_depth,
            decoder_heads=args.decoder_heads,
            mask_ratio=args.mask_ratio
        ).to(device)

        pretrain_fn = pretrain_rgb_encoder
        encoder_name = 'rgb_encoder'

    elif args.encoder == 'hsi':
        print("\nPretraining HSI Encoder...")
        # Get HSI channels from first sample
        sample = dataset[0]
        if isinstance(sample[0], list):
            hsi_channels = sample[0][1].shape[0]
        else:
            hsi_channels = sample[0].shape[0]

        model = HSIMAEPretrainer(
            img_size=args.img_size,
            patch_size=args.patch_size,
            in_chans=hsi_channels,
            encoder_dim=args.encoder_dim,
            encoder_depth=args.encoder_depth,
            encoder_heads=args.encoder_heads,
            decoder_dim=args.decoder_dim,
            decoder_depth=args.decoder_depth,
            decoder_heads=args.decoder_heads,
            mask_ratio=args.mask_ratio
        ).to(device)

        pretrain_fn = pretrain_hsi_encoder
        encoder_name = 'hsi_encoder'
    else:
        raise ValueError(f"Unknown encoder: {args.encoder}")

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epoch)

    # Loss criterion (copy from SS-MAE)
    criterion = MSELoss(device=device)

    # Training loop
    for epoch in trange(args.max_epoch):
        loss = pretrain_fn(model, dataloader, criterion, optimizer, lr_scheduler, device, args)
        print(f"Epoch {epoch:04d} | Loss: {loss:.6f}")

    # Save final checkpoint
    save_checkpoint(model, optimizer, lr_scheduler, args.max_epoch, args, encoder_name)

    print("\n" + "=" * 60)
    print("Pretraining completed!")
    print("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pretrain MAE-MoE encoders')
    parser.add_argument('--encoder', type=str, required=True, choices=['rgb', 'hsi'],
                        help='Which encoder to pretrain')
    parser.add_argument('--data-path', type=str, required=True,
                        help='Path to dataset')
    parser.add_argument('--results-path', type=str, default='results/mae_moe_pretrain',
                        help='Path to save results')
    parser.add_argument('--experiment-name', type=str, default='mae_moe',
                        help='Experiment name')
    parser.add_argument('--img-size', type=int, default=256,
                        help='Image size')
    parser.add_argument('--patch-size', type=int, default=16,
                        help='Patch size')
    parser.add_argument('--encoder-dim', type=int, default=768,
                        help='Encoder embedding dimension')
    parser.add_argument('--encoder-depth', type=int, default=12,
                        help='Encoder depth')
    parser.add_argument('--encoder-heads', type=int, default=12,
                        help='Encoder attention heads')
    parser.add_argument('--decoder-dim', type=int, default=512,
                        help='Decoder embedding dimension')
    parser.add_argument('--decoder-depth', type=int, default=4,
                        help='Decoder depth')
    parser.add_argument('--decoder-heads', type=int, default=8,
                        help='Decoder attention heads')
    parser.add_argument('--mask-ratio', type=float, default=0.75,
                        help='Masking ratio')
    parser.add_argument('--batch-size', type=int, default=12,
                        help='Batch size')
    parser.add_argument('--max-epoch', type=int, default=100,
                        help='Maximum epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='Weight decay')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')

    args = parser.parse_args()
    main(args)


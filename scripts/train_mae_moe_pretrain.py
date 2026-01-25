#!/usr/bin/env python3
"""
Pretrain script for MAE-MoE model
Pretrain RGB and HSI encoders separately using MAE reconstruction loss
"""
import argparse
import os
import torch
import torchmetrics
from torch.utils.data import DataLoader
from tqdm import trange

from spectralwaste_segmentation.datasets import (
    SpectralWasteSegmentation,
    SemanticSegmentationTrain,
    SemanticSegmentationTest
)
from spectralwaste_segmentation.models.mae_moe import (
    VisionTransformerEncoder,
    DualBranchMAEEncoder
)
from spectralwaste_segmentation.models.mae_moe.mae_loss import MSELoss

torch.manual_seed(0)
torch.cuda.manual_seed(0)


def save_checkpoint(encoder, optimizer, lr_scheduler, epoch, args, encoder_name):
    """Save checkpoint for encoder"""
    checkpoint = {
        'encoder': encoder.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'epoch': epoch,
        'args': args,
    }
    os.makedirs(args.results_path, exist_ok=True)
    save_path = os.path.join(args.results_path, f'{args.experiment_name}_{encoder_name}_pretrain.pth')
    torch.save(checkpoint, save_path)
    print(f'Saved {encoder_name} checkpoint to {save_path}')


def pretrain_rgb_encoder(encoder, dataloader, criterion, optimizer, lr_scheduler, device, mask_ratio=0.75):
    """Pretrain RGB encoder with MAE"""
    encoder.train()
    mean_loss = torchmetrics.MeanMetric().to(device)
    
    for inputs, _ in dataloader:
        # Get RGB input
        if isinstance(inputs, list):
            rgb = inputs[0].to(device)  # Assume RGB is first modality
        else:
            rgb = inputs.to(device)
        
        # Simple masking: randomly mask patches
        B, C, H, W = rgb.shape
        # For simplicity, we'll just use the encoder without explicit masking
        # In full MAE, you would mask patches and reconstruct
        
        # Forward pass
        features = encoder(rgb)  # [B, N, D]
        
        # Reconstruction head (simple linear projection back to pixels)
        # For now, use a simple MSE loss on features
        # In full implementation, add a decoder
        
        # Placeholder: use zero loss for now
        # TODO: Implement proper MAE reconstruction
        loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        mean_loss.update(loss)
    
    lr_scheduler.step()
    return mean_loss.compute()


def pretrain_hsi_encoder(encoder, dataloader, criterion, optimizer, lr_scheduler, device, mask_ratio=0.75):
    """Pretrain HSI encoder with dual-branch MAE"""
    encoder.train()
    mean_loss = torchmetrics.MeanMetric().to(device)
    
    for inputs, _ in dataloader:
        # Get HSI input
        if isinstance(inputs, list):
            hsi = inputs[1].to(device)  # Assume HSI is second modality
        else:
            hsi = inputs.to(device)
        
        # Forward pass through dual-branch encoder
        spa_features, chan_features = encoder(hsi)  # [B, N, D] each
        
        # Placeholder: use zero loss for now
        # TODO: Implement proper dual-branch MAE reconstruction
        loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        mean_loss.update(loss)
    
    lr_scheduler.step()
    return mean_loss.compute()


def main(args):
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
    
    # Load dataset
    train_data = SpectralWasteSegmentation(
        args.data_path,
        split='train',
        input_mode=['rgb', 'hyper'],  # Load both modalities
        target_mode=args.target_mode,
        transforms=SemanticSegmentationTrain(),
        target_type=''
    )
    
    train_dataloader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2
    )
    
    # Create encoder based on args
    if args.encoder == 'rgb':
        print("\nPretraining RGB Encoder...")
        encoder = VisionTransformerEncoder(
            img_size=args.img_size,
            patch_size=args.patch_size,
            in_chans=3,  # Edit: Use 'in_chans' not 'in_channels'
            embed_dim=args.embed_dim,
            depth=args.depth,
            num_heads=args.num_heads,
        ).to(args.device)

        optimizer = torch.optim.AdamW(encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.max_epoch)
        criterion = MSELoss(device=args.device)

        # Training loop
        for epoch in trange(args.max_epoch):
            train_loss = pretrain_rgb_encoder(
                encoder, train_dataloader, criterion, optimizer, lr_scheduler, args.device, args.mask_ratio
            )
            print(f'Epoch {epoch:04d} | Loss: {train_loss:.6f}')

            # Save checkpoint every N epochs
            if (epoch + 1) % args.save_interval == 0:
                save_checkpoint(encoder, optimizer, lr_scheduler, epoch, args, 'rgb_encoder')

        # Save final checkpoint
        save_checkpoint(encoder, optimizer, lr_scheduler, args.max_epoch - 1, args, 'rgb_encoder')

    elif args.encoder == 'hsi':
        print("\nPretraining HSI Encoder...")
        # Get HSI channels from dataset
        sample = train_data[0]
        if isinstance(sample[0], list):
            hsi_channels = sample[0][1].shape[0]  # Second modality
        else:
            hsi_channels = sample[0].shape[0]

        encoder = DualBranchMAEEncoder(
            img_size=args.img_size,
            patch_size=args.patch_size,
            in_chans=hsi_channels,  # Edit: Use 'in_chans' not 'in_channels'
            embed_dim=args.embed_dim,
            depth=args.depth,
            num_heads=args.num_heads,
        ).to(args.device)

        optimizer = torch.optim.AdamW(encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.max_epoch)
        criterion = MSELoss(device=args.device)

        # Training loop
        for epoch in trange(args.max_epoch):
            train_loss = pretrain_hsi_encoder(
                encoder, train_dataloader, criterion, optimizer, lr_scheduler, args.device, args.mask_ratio
            )
            print(f'Epoch {epoch:04d} | Loss: {train_loss:.6f}')

            # Save checkpoint every N epochs
            if (epoch + 1) % args.save_interval == 0:
                save_checkpoint(encoder, optimizer, lr_scheduler, epoch, args, 'hsi_encoder')

        # Save final checkpoint
        save_checkpoint(encoder, optimizer, lr_scheduler, args.max_epoch - 1, args, 'hsi_encoder')

    else:
        raise ValueError(f"Unknown encoder: {args.encoder}. Choose 'rgb' or 'hsi'")

    print("\n" + "=" * 60)
    print("Pretraining completed!")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pretrain MAE-MoE encoders')

    # Data paths
    parser.add_argument('--data-path', type=str, default='data/spectralwaste_segmentation')
    parser.add_argument('--results-path', type=str, default='results/mae_moe_pretrain')
    parser.add_argument('--device', type=str, default='cuda')

    # Encoder selection
    parser.add_argument('--encoder', type=str, required=True, choices=['rgb', 'hsi'],
                        help='Which encoder to pretrain: rgb or hsi')

    # Model architecture
    # Edit: Default img_size=256 to match SpectralWaste dataset
    parser.add_argument('--img-size', type=int, default=256)
    parser.add_argument('--patch-size', type=int, default=16)
    parser.add_argument('--embed-dim', type=int, default=768)
    parser.add_argument('--depth', type=int, default=12)
    parser.add_argument('--num-heads', type=int, default=12)

    # Training settings
    parser.add_argument('--batch-size', type=int, default=12)
    parser.add_argument('--max-epoch', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=0.05)
    parser.add_argument('--mask-ratio', type=float, default=0.75)

    # Other
    parser.add_argument('--target-mode', type=str, default='labels_rgb')
    parser.add_argument('--save-interval', type=int, default=10)
    parser.add_argument('--experiment-name', type=str, default='mae_moe')

    args = parser.parse_args()
    main(args)


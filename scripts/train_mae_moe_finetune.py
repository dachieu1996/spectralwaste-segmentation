#!/usr/bin/env python3
"""
Finetune script for MAE-MoE model
Train full MAE-MoE model on segmentation task with optional pretrained encoders
"""
import argparse
import os
import torch
import torchmetrics
from torch.utils.data import DataLoader
from tqdm import trange
import wandb

from spectralwaste_segmentation.datasets import (
    SpectralWasteSegmentation,
    SemanticSegmentationTrain,
    SemanticSegmentationTest
)
from spectralwaste_segmentation.models.mae_moe import MAEMoE

torch.manual_seed(0)
torch.cuda.manual_seed(0)


def save_checkpoint(model, optimizer, lr_scheduler, epoch, args, suffix):
    """Save checkpoint"""
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'epoch': epoch,
        'args': args,
    }
    os.makedirs(args.results_path, exist_ok=True)
    save_path = os.path.join(args.results_path, f'{args.experiment_name}.{suffix}.pth')
    torch.save(checkpoint, save_path)
    print(f'Saved checkpoint to {save_path}')


def median_frequency_exp(dataset, num_classes, c=1.02):
    """Calculate median frequency balancing weights"""
    _, target = dataset[0]
    label_freq = torch.zeros(num_classes)
    
    for i in range(len(dataset)):
        _, target = dataset[i]
        for c_i in range(num_classes):
            label_freq[c_i] += (target == c_i).sum()
    
    label_freq = label_freq / label_freq.sum()
    median_freq = torch.median(label_freq[label_freq != 0])
    weights = median_freq / label_freq
    weights = torch.exp(c * weights)
    weights[label_freq == 0] = 0
    
    return weights


def train_epoch(model, dataloader, criterion, optimizer, lr_scheduler, device, loss_coef=1e-2):
    """Train for one epoch
    
    Args:
        loss_coef: Coefficient for MoE load balancing loss
    """
    model.train()
    mean_loss = torchmetrics.MeanMetric().to(device)
    mean_seg_loss = torchmetrics.MeanMetric().to(device)
    mean_aux_loss = torchmetrics.MeanMetric().to(device)
    
    for inputs, target in dataloader:
        # inputs is a list [rgb, hsi]
        if not isinstance(inputs, list):
            raise ValueError("MAE-MoE requires multimodal input [rgb, hsi]")
        
        rgb = inputs[0].to(device)
        hsi = inputs[1].to(device)
        target = target.to(device)
        
        # Forward pass
        output, aux_loss = model(rgb, hsi, train=True)
        
        # Segmentation loss
        seg_loss = criterion(output, target)
        
        # Total loss = segmentation loss + load balancing loss
        total_loss = seg_loss + loss_coef * aux_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        mean_loss.update(total_loss)
        mean_seg_loss.update(seg_loss)
        mean_aux_loss.update(aux_loss)
    
    lr_scheduler.step()
    return mean_loss.compute(), mean_seg_loss.compute(), mean_aux_loss.compute()


def evaluate(model, dataloader, criterion, num_classes, device):
    """Evaluate model"""
    model.eval()
    mean_loss = torchmetrics.MeanMetric().to(device)
    class_iou = torchmetrics.JaccardIndex(num_classes=num_classes, task='multiclass', average='none').to(device)
    
    with torch.inference_mode():
        for inputs, target in dataloader:
            if not isinstance(inputs, list):
                raise ValueError("MAE-MoE requires multimodal input [rgb, hsi]")
            
            rgb = inputs[0].to(device)
            hsi = inputs[1].to(device)
            target = target.to(device)
            
            # Forward pass (no aux_loss in eval mode)
            output = model(rgb, hsi, train=False)
            
            mean_loss.update(criterion(output, target))
            class_iou.update(output, target)
    
    mean_loss = mean_loss.compute()
    class_iou = class_iou.compute()
    miou = class_iou[1:].mean()
    iou_std = class_iou[1:].std()
    return mean_loss, class_iou, miou, iou_std


def main(args):
    print("=" * 60)
    print("MAE-MoE Finetuning")
    print("=" * 60)
    print(f"Data path: {args.data_path}")
    print(f"Results path: {args.results_path}")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max epochs: {args.max_epoch}")
    print(f"Router type: {args.router_type}")
    print(f"Num experts: {args.num_experts}")
    print(f"Top-k: {args.top_k}")
    print(f"Loss coefficient: {args.loss_coef}")
    if args.rgb_encoder_pretrain:
        print(f"RGB encoder pretrain: {args.rgb_encoder_pretrain}")
    if args.hsi_encoder_pretrain:
        print(f"HSI encoder pretrain: {args.hsi_encoder_pretrain}")
    print("=" * 60)

    # Load datasets
    train_data = SpectralWasteSegmentation(
        args.data_path,
        split='train',
        input_mode=['rgb', 'hyper'],
        target_mode=args.target_mode,
        transforms=SemanticSegmentationTrain(),
        target_type=''
    )
    val_data = SpectralWasteSegmentation(
        args.data_path,
        split='val',
        input_mode=['rgb', 'hyper'],
        target_mode=args.target_mode,
        transforms=SemanticSegmentationTest(),
        target_type=''
    )
    test_data = SpectralWasteSegmentation(
        args.data_path,
        split='test',
        input_mode=['rgb', 'hyper'],
        target_mode=args.target_mode,
        transforms=SemanticSegmentationTest(),
        target_type=''
    )

    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_dataloader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Get HSI channels from dataset
    sample = train_data[0]
    hsi_channels = sample[0][1].shape[0]  # Second modality (HSI)

    # Create model
    model = MAEMoE(
        img_size=args.img_size,
        patch_size=args.patch_size,
        rgb_channels=3,
        hsi_channels=hsi_channels,
        num_classes=train_data.num_classes,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        num_experts=args.num_experts,
        top_k=args.top_k,
        router_type=args.router_type,
        gating=args.gating,
        noisy_gating=args.noisy_gating,
    ).to(args.device)

    # Load pretrained encoders if provided
    if args.rgb_encoder_pretrain:
        print(f"\nLoading pretrained RGB encoder from {args.rgb_encoder_pretrain}")
        checkpoint = torch.load(args.rgb_encoder_pretrain, map_location=args.device)
        model.rgb_encoder.load_state_dict(checkpoint['encoder'], strict=False)
        print("✓ RGB encoder loaded")

    if args.hsi_encoder_pretrain:
        print(f"\nLoading pretrained HSI encoder from {args.hsi_encoder_pretrain}")
        checkpoint = torch.load(args.hsi_encoder_pretrain, map_location=args.device)
        model.hsi_encoder.load_state_dict(checkpoint['encoder'], strict=False)
        print("✓ HSI encoder loaded")

    # Create optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, args.max_epoch, 0.9)

    # Calculate loss weights
    loss_weights = median_frequency_exp(train_data, train_data.num_classes, 0.12)
    criterion = torch.nn.CrossEntropyLoss(loss_weights.to(args.device))

    # Resume from checkpoint if provided
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=args.device)
        model.load_state_dict(checkpoint['model'])
        if not args.test_only:
            args.start_epoch = checkpoint['epoch'] + 1
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        print("✓ Checkpoint loaded")

    # Test only mode
    if args.test_only:
        print("\nEvaluating model on test set...")
        test_loss, test_class_iou, test_miou, test_iou_std = evaluate(
            model, test_dataloader, criterion, train_data.num_classes, args.device
        )
        print(f'Test Loss: {test_loss:.4f}')
        print(f'Test mIoU: {test_miou:.4f}')
        print(f'Test IoU std: {test_iou_std:.4f}')
        print(f'Test class IoU: {test_class_iou.tolist()}')
        return

    # Start logging
    if args.wandb:
        wandb.init(project=args.wandb, entity='separa', name=args.experiment_name, config=args)

    # Training loop
    best_val_miou = 0
    best_model = model

    for epoch in trange(args.start_epoch, args.max_epoch):
        train_loss, train_seg_loss, train_aux_loss = train_epoch(
            model, train_dataloader, criterion, optimizer, lr_scheduler, args.device, args.loss_coef
        )
        val_loss, val_class_iou, val_miou, val_iou_std = evaluate(
            model, val_dataloader, criterion, train_data.num_classes, args.device
        )

        if val_miou > best_val_miou:
            best_val_miou = val_miou
            save_checkpoint(model, optimizer, lr_scheduler, epoch, args, 'best')
            best_model = model

        print(f'Epoch {epoch:04d} | Train Loss: {train_loss:.4f} (Seg: {train_seg_loss:.4f}, Aux: {train_aux_loss:.6f}) | Val Loss: {val_loss:.4f} | Val mIoU: {val_miou:.4f}')

        if args.wandb:
            val_class_iou_dict = {f'val/iou_{train_data.classes_names[i]}': val_class_iou[i] for i in range(train_data.num_classes)}
            wandb.log({
                "train/lr": lr_scheduler.get_last_lr()[0],
                "train/loss": train_loss,
                "train/seg_loss": train_seg_loss,
                "train/aux_loss": train_aux_loss,
                "val/loss": val_loss,
                "val/miou": val_miou,
                "val/iou_std": val_iou_std,
                **val_class_iou_dict
            })

    save_checkpoint(model, optimizer, lr_scheduler, args.max_epoch - 1, args, 'last')

    # Evaluate best model on test set
    print("\nEvaluating best model on test set...")
    test_loss, test_class_iou, test_miou, test_iou_std = evaluate(
        best_model, test_dataloader, criterion, train_data.num_classes, args.device
    )
    print(f'Test Loss: {test_loss:.4f}')
    print(f'Test mIoU: {test_miou:.4f}')
    print(f'Test IoU std: {test_iou_std:.4f}')

    if args.wandb:
        test_class_iou_dict = {f'test/best_iou_{train_data.classes_names[i]}': test_class_iou[i] for i in range(train_data.num_classes)}
        wandb.log({
            "test/best_loss": test_loss,
            "test/best_miou": test_miou,
            "test/best_iou_std": test_iou_std,
            **test_class_iou_dict
        })

    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Finetune MAE-MoE for segmentation')

    # Data paths
    parser.add_argument('--data-path', type=str, default='data/spectralwaste_segmentation')
    parser.add_argument('--results-path', type=str, default='results/mae_moe_finetune')
    parser.add_argument('--device', type=str, default='cuda')

    # Model architecture
    # Edit: Default img_size=256 to match SpectralWaste dataset
    parser.add_argument('--img-size', type=int, default=256)
    parser.add_argument('--patch-size', type=int, default=16)
    parser.add_argument('--embed-dim', type=int, default=768)
    parser.add_argument('--depth', type=int, default=12)
    parser.add_argument('--num-heads', type=int, default=12)

    # MoE settings
    parser.add_argument('--num-experts', type=int, default=8)
    parser.add_argument('--top-k', type=int, default=4)
    parser.add_argument('--router-type', type=str, default='permod', choices=['joint', 'permod', 'disjoint'])
    parser.add_argument('--gating', type=str, default='softmax', choices=['softmax', 'laplace', 'gaussian'])
    parser.add_argument('--noisy-gating', action='store_true', default=True)
    parser.add_argument('--loss-coef', type=float, default=1e-2, help='Coefficient for MoE load balancing loss')

    # Training settings
    parser.add_argument('--batch-size', type=int, default=12)
    parser.add_argument('--start-epoch', type=int, default=0)
    parser.add_argument('--max-epoch', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=0.05)

    # Pretrained encoders
    parser.add_argument('--rgb-encoder-pretrain', type=str, default='', help='Path to pretrained RGB encoder')
    parser.add_argument('--hsi-encoder-pretrain', type=str, default='', help='Path to pretrained HSI encoder')

    # Other
    parser.add_argument('--target-mode', type=str, default='labels_rgb')
    parser.add_argument('--resume', type=str, default='', help='Path to checkpoint to resume from')
    parser.add_argument('--test-only', action='store_true')
    parser.add_argument('--experiment-name', type=str, default='mae_moe_finetune')
    parser.add_argument('--wandb', type=str, default='', help='Wandb project name')

    args = parser.parse_args()
    main(args)


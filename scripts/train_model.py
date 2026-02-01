import argparse
import os
import uuid

import torch
import torchmetrics

from torch.utils.data import Dataset, DataLoader

from tqdm import trange
import wandb

from spectralwaste_segmentation.datasets import (
    SpectralWasteSegmentation,
    HeterogeneousSpectralWasteSegmentation,
    SemanticSegmentationTest,
    SemanticSegmentationTrain
)
from spectralwaste_segmentation import models

torch.manual_seed(0)
torch.cuda.manual_seed(0)


def save_checkpoint(model, optimizer, lr_scheduler, epoch, args, suffix):
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'epoch': epoch,
        'args': args,
    }
    torch.save(checkpoint, os.path.join(args.results_path, f'{args.experiment_name}.{suffix}.pth'))

def median_frequency_exp(dataset: Dataset, num_classes: int, soft: float):
    # Process the dataset in parallel
    loader = DataLoader(dataset, batch_size=64, num_workers=2, shuffle=False)

    # Initialize counts
    classes_freqs = torch.zeros(num_classes, dtype=torch.int64)

    for _, target in loader:
        classes, counts = torch.unique(target, return_counts=True)
        ignore = torch.bitwise_or(classes < 0, classes >= num_classes)
        classes_freqs.index_add_(0, classes[~ignore], counts[~ignore])

    zeros = classes_freqs == 0
    if zeros.sum() != 0:
        print("There are some classes not present in the training samples")

    result = classes_freqs.median() / classes_freqs
    result[zeros] = 0  # avoid inf values
    return result ** soft

def train_epoch(model, dataloader, criterion, optimizer, lr_scheduler, device):
    model.train()
    mean_loss = torchmetrics.MeanMetric().to(device)

    for input, target in dataloader:
        if isinstance(input, list):
            input = [i.to(device) for i in input]
        else:
            input = input.to(device)
        target = target.to(device)

        output = model(input)
        aux_loss = None
        if isinstance(output, tuple):
            output, aux_loss = output

        loss = criterion(output, target)
        if aux_loss is not None:
            loss = loss + aux_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        mean_loss.update(loss)

    lr_scheduler.step()
    return mean_loss.compute()

def evaluate(model, dataloader, criterion, num_classes, device):
    model.eval()
    mean_loss = torchmetrics.MeanMetric().to(device)
    class_iou = torchmetrics.JaccardIndex(num_classes=num_classes, task='multiclass', average='none').to(device)

    with torch.inference_mode():
        for input, target in dataloader:
            if isinstance(input, list):
                input = [i.to(device) for i in input]
            else:
                input = input.to(device)
            target = target.to(device)

            output = model(input)
            aux_loss = None
            if isinstance(output, tuple):
                output, aux_loss = output

            loss = criterion(output, target)
            if aux_loss is not None:
                loss = loss + aux_loss
            mean_loss.update(loss)
            class_iou.update(output, target)

    mean_loss = mean_loss.compute()
    class_iou = class_iou.compute()
    miou = class_iou[1:].mean()
    iou_std = class_iou[1:].std()
    return mean_loss, class_iou, miou, iou_std

def main(args):
    suffix = f'.hetero' if args.hetero else ''
    args.experiment_name = f'{args.model}.{args.input_mode}.{args.target_mode}{suffix}.{str(uuid.uuid4())[:4]}'
    print(args.experiment_name)

    def _normalize_input_mode(mode: str) -> str:
        mode = mode.strip().lower()
        if mode == "hsi":
            return "hyper"
        return mode

    input_mode = args.input_mode
    if ',' in input_mode:
        input_mode = [_normalize_input_mode(m) for m in input_mode.split(',')]
    else:
        input_mode = _normalize_input_mode(input_mode)

    def _parse_modal_mix(spec: str) -> dict[str, float]:
        # "both:0.4,rgb:0.3,hyper:0.3"
        mix = {"both": 0.0, "rgb": 0.0, "hyper": 0.0}
        if not spec:
            return mix
        for part in spec.split(','):
            k, v = part.split(':', maxsplit=1)
            k = k.strip().lower()
            v = float(v.strip())
            if k == "hsi":
                k = "hyper"
            if k not in mix:
                raise ValueError(f"Unknown modal_mix key: {k} (expected one of {sorted(mix)})")
            mix[k] = v
        return mix

    if args.hetero:
        if not isinstance(input_mode, list) or len(input_mode) != 2:
            raise ValueError("--hetero requires --input-mode to be a 2-modality list, e.g. 'rgb,hyper'.")
        if set(input_mode) != {"rgb", "hyper"}:
            raise ValueError("--hetero currently supports only 'rgb' and 'hyper' (HSI).")

        train_data = HeterogeneousSpectralWasteSegmentation(
            args.data_path,
            split='train',
            input_mode=input_mode,
            target_mode=args.target_mode,
            transforms=SemanticSegmentationTrain(),
            target_type='',
            modal_mix=_parse_modal_mix(args.modal_mix),
        )
        val_data = HeterogeneousSpectralWasteSegmentation(
            args.data_path,
            split='val',
            input_mode=input_mode,
            target_mode=args.target_mode,
            transforms=SemanticSegmentationTest(),
            target_type='',
            forced_mode="both",
        )
        test_data = HeterogeneousSpectralWasteSegmentation(
            args.data_path,
            split='test',
            input_mode=input_mode,
            target_mode=args.target_mode,
            transforms=SemanticSegmentationTest(),
            target_type='',
            forced_mode="both",
        )
    else:
        train_data = SpectralWasteSegmentation(args.data_path, split='train', input_mode=input_mode, target_mode=args.target_mode, transforms=SemanticSegmentationTrain(), target_type='')
        val_data = SpectralWasteSegmentation(args.data_path, split='val', input_mode=input_mode, target_mode=args.target_mode, transforms=SemanticSegmentationTest(), target_type='')
        test_data = SpectralWasteSegmentation(args.data_path, split='test', input_mode=input_mode, target_mode=args.target_mode, transforms=SemanticSegmentationTest(), target_type='')

    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_dataloader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=2)

    if isinstance(train_data.num_channels, list):
        list_models = {"mininet_multimodal", "segformer_b0_multimodal", "cmx_b0", "unet_multimodal", "unet_moe_multimodal"}
        if args.model not in list_models:
            raise ValueError(f"--input-mode produces multiple tensors but --model={args.model} expects a single tensor. Try one of: {sorted(list_models)}")

    model = models.create_model(args.model, train_data.num_channels, train_data.num_classes).to(args.device)
    if args.moe_aux_loss and hasattr(model, "return_aux_loss"):
        model.return_aux_loss = True
    optimizer, lr_scheduler = models.create_optimizers(args.model, model, args.max_epoch)

    # Calculate loss weights and define loss
    loss_weights = median_frequency_exp(train_data, train_data.num_classes, 0.12)
    criterion = torch.nn.CrossEntropyLoss(loss_weights.to(args.device))

    if args.resume:
        try:
            with torch.serialization.safe_globals([argparse.Namespace]):
                checkpoint = torch.load(args.resume, map_location=args.device)
        except AttributeError:
            checkpoint = torch.load(args.resume, map_location=args.device)
        model.load_state_dict(checkpoint['model'])
        if not args.test_only:
            args.start_epoch = checkpoint['epoch'] + 1
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

    if args.test_only:
        # Evaluate model
        test_loss, test_class_iou, test_miou, test_iou_std = evaluate(model, test_dataloader, criterion, train_data.num_classes, args.device)
        print(f'test/loss: {test_loss} | test/class_iou: {test_class_iou.tolist()} | test/miou: {test_miou}, test/iou_std: {test_iou_std}')
        return

    # Start logging
    if args.wandb:
        wandb.init(project=args.wandb, entity='separa', name=args.experiment_name, config=args)

    # Train
    os.makedirs(args.results_path, exist_ok=True)
    best_val_miou = 0
    best_path = os.path.join(args.results_path, f'{args.experiment_name}.best.pth')

    val_dataloaders_by_mode = None
    test_dataloaders_by_mode = None
    if args.eval_per_mode and args.hetero:
        val_dataloaders_by_mode = {
            m: DataLoader(
                HeterogeneousSpectralWasteSegmentation(
                    args.data_path,
                    split='val',
                    input_mode=input_mode,
                    target_mode=args.target_mode,
                    transforms=SemanticSegmentationTest(),
                    target_type='',
                    forced_mode=m,
                ),
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=2,
            )
            for m in ("both", "rgb", "hyper")
        }
        test_dataloaders_by_mode = {
            m: DataLoader(
                HeterogeneousSpectralWasteSegmentation(
                    args.data_path,
                    split='test',
                    input_mode=input_mode,
                    target_mode=args.target_mode,
                    transforms=SemanticSegmentationTest(),
                    target_type='',
                    forced_mode=m,
                ),
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=2,
            )
            for m in ("both", "rgb", "hyper")
        }

    for epoch in trange(args.start_epoch, args.max_epoch):
        train_loss = train_epoch(model, train_dataloader, criterion, optimizer, lr_scheduler, args.device)
        val_loss, val_class_iou, val_miou, val_iou_std = evaluate(model, val_dataloader, criterion, train_data.num_classes, args.device)
        val_miou_f = float(val_miou)

        if val_miou_f > best_val_miou:
            save_checkpoint(model, optimizer, lr_scheduler, epoch, args, 'best')
            best_val_miou = val_miou_f

        print(f'epoch: {epoch:04d} | train/loss: {train_loss:.4f} | val/loss: {val_loss:.4f} | val/miou: {val_miou:.4f}')

        if args.wandb:
            val_class_iou = {f'val/iou_{train_data.classes_names[i]}': val_class_iou[i] for i in range(train_data.num_classes)}
            wandb.log({
                "train/lr": lr_scheduler.get_last_lr()[0],
                "train/loss": train_loss,
                "val/loss": val_loss,
                "val/miou": val_miou,
                "val/iou_std": val_iou_std,
                **val_class_iou
            })

        if val_dataloaders_by_mode is not None:
            by_mode = {}
            for m, dl in val_dataloaders_by_mode.items():
                _, _, miou, _ = evaluate(model, dl, criterion, train_data.num_classes, args.device)
                by_mode[m] = float(miou)
            print(f"val/miou by mode: {by_mode}")
            if args.wandb:
                wandb.log({f"val/miou_{k}": v for k, v in by_mode.items()})

    save_checkpoint(model, optimizer, lr_scheduler, epoch, args, 'last')

    # Evaluate the best checkpoint (if available)
    if os.path.exists(best_path):
        try:
            with torch.serialization.safe_globals([argparse.Namespace]):
                checkpoint = torch.load(best_path, map_location=args.device)
        except AttributeError:
            checkpoint = torch.load(best_path, map_location=args.device)
        model.load_state_dict(checkpoint["model"])

    test_loss, test_class_iou, test_miou, test_iou_std = evaluate(model, test_dataloader, criterion, train_data.num_classes, args.device)
    print(f'test/loss: {test_loss} | test/miou: {test_miou}')

    if test_dataloaders_by_mode is not None:
        by_mode = {}
        for m, dl in test_dataloaders_by_mode.items():
            _, _, miou, _ = evaluate(model, dl, criterion, train_data.num_classes, args.device)
            by_mode[m] = float(miou)
        print(f"test/miou by mode: {by_mode}")

    if args.wandb:
        test_class_iou = {f'test/best_iou_{train_data.classes_names[i]}': test_class_iou[i] for i in range(train_data.num_classes)}
        wandb.log({
            "test/best_loss": test_loss,
            "test/best_miou": test_miou,
            "test/best_iou_std": test_iou_std,
            **test_class_iou
        })
        if test_dataloaders_by_mode is not None:
            wandb.log({f"test/miou_{k}": v for k, v in by_mode.items()})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # data paths
    parser.add_argument('--data-path', type=str, default='data/spectralwaste_segmentation')
    parser.add_argument('--results-path', type=str, default='results')
    parser.add_argument('--device', type=str, default='cuda')
    # setting
    parser.add_argument('--model', type=str, default='mininet')
    parser.add_argument('--input-mode', type=str, default='rgb')
    parser.add_argument('--target-mode', type=str, default='labels_rgb')
    # heterogeneous training (single model, mixed modalities)
    parser.add_argument('--hetero', action='store_true', help="Train with mixed RGB/HSI availability (zeros-out missing modality). Requires --input-mode 'rgb,hyper' (or 'rgb,hsi').")
    parser.add_argument('--modal-mix', type=str, default='both:0.34,rgb:0.33,hyper:0.33', help="Mix for --hetero. Format: 'both:0.4,rgb:0.3,hyper:0.3'.")
    parser.add_argument('--eval-per-mode', action='store_true', help="When --hetero, also report val/test mIoU for modes: both/rgb/hyper.")
    parser.add_argument('--moe-aux-loss', action='store_true', help="If supported by the model (e.g. UNetMoE), add the MoE load-balancing auxiliary loss.")
    # training
    parser.add_argument('--batch-size', type=int, default=12)
    parser.add_argument('--start-epoch', type=int, default=0)
    parser.add_argument('--max-epoch', type=int, default=200)
    parser.add_argument('--resume', type=str, default='', help='Path of a checkpoint')
    parser.add_argument('--test-only', action='store_true')
    parser.add_argument('--wandb', type=str, default='', help='W&B project name')

    args = parser.parse_args()
    main(args)

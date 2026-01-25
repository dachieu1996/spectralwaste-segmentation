import torch

from . import mininet
from . import cmx
from . import segformer
from . import fusemoe
from . import mae_moe


class MiniNet(torch.nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.model = mininet.MiniNetv2(in_channels, num_classes, interpolate=True)
        self.num_classes = num_classes

    def forward(self, input):
        return self.model(input)


class MiniNetMultimodal(torch.nn.Module):
    def __init__(self, in_channels: list[int], num_classes: int):
        super().__init__()
        self.model = mininet.MiniNetv2(sum(in_channels), num_classes, interpolate=True)

    def forward(self, inputs):
        input = torch.concat(inputs, axis=1)
        return self.model(input)


class SegFormer(torch.nn.Module):
    def __init__(self, in_channels: int, num_classes: int, encoder: str):
        super().__init__()
        self.model = segformer.SegFormer(in_chans=in_channels, num_classes=num_classes, backbone=encoder)

    def forward(self, input):
        return self.model(input)


class SegFormerMultimodal(torch.nn.Module):
    def __init__(self, in_channels: int, num_classes: int, encoder: str):
        super().__init__()
        self.model = segformer.SegFormer(in_chans=sum(in_channels), num_classes=num_classes, backbone=encoder)

    def forward(self, inputs):
        input = torch.concat(inputs, axis=1)
        return self.model(input)


class CMX(torch.nn.Module):
    def __init__(self, in_channels: list[int], num_classes: int, encoder: str):
        super().__init__()
        assert len(in_channels) == 2 and in_channels[0] == 3
        self.model = cmx.EncoderDecoder(extra_in_chans=in_channels[1], num_classes=num_classes, encoder=encoder)

    def forward(self, inputs):
        input1, input2 = inputs
        return self.model(input1, input2)


class FuseMoE(torch.nn.Module):
    def __init__(self, in_channels: int | list[int], num_classes: int):
        super().__init__()
        self.model = fusemoe.FuseMoESegmentation(in_channels, num_classes)

    def forward(self, inputs):
        return self.model(inputs)


class MAEMoE(torch.nn.Module):
    def __init__(self, in_channels: list[int], num_classes: int,
                 img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12,
                 num_experts=8, top_k=4, router_type='permod', gating='softmax', noisy_gating=True):
        super().__init__()
        assert len(in_channels) == 2 and in_channels[0] == 3, "MAE-MoE requires [rgb_channels=3, hsi_channels]"

        self.model = mae_moe.MAEMoE(
            img_size=img_size,
            patch_size=patch_size,
            rgb_channels=in_channels[0],
            hsi_channels=in_channels[1],
            num_classes=num_classes,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            num_experts=num_experts,
            top_k=top_k,
            router_type=router_type,
            gating=gating,
            noisy_gating=noisy_gating,
        )
        self.training_mode = True  # Track if we're in training mode

    def forward(self, inputs):
        """Forward pass

        Args:
            inputs: List of [rgb, hsi] tensors

        Returns:
            output: Segmentation output [B, num_classes, H, W]
            aux_loss: MoE load balancing loss (only in training mode)
        """
        rgb, hsi = inputs
        if self.training:
            output, aux_loss = self.model(rgb, hsi, train=True)
            return output, aux_loss
        else:
            output = self.model(rgb, hsi, train=False)
            return output


def create_model(name, in_channels, num_classes):
    if name == 'mininet':
        model = MiniNet(in_channels, num_classes)
    elif name == 'mininet_multimodal':
        model = MiniNetMultimodal(in_channels, num_classes)
    elif name == 'segformer_b0':
        model = SegFormer(in_channels, num_classes, 'mit_b0')
    elif name == 'segformer_b0_multimodal':
        model = SegFormerMultimodal(in_channels, num_classes, 'mit_b0')
    elif name == 'cmx_b0':
        model = CMX(in_channels, num_classes, 'mit_b0')
    elif name == 'fusemoe':
        model = FuseMoE(in_channels, num_classes)
    elif name == 'mae_moe':
        model = MAEMoE(in_channels, num_classes)
    elif name == 'mae_moe_small':
        model = MAEMoE(in_channels, num_classes, embed_dim=384, depth=6, num_heads=6, num_experts=4, top_k=2)
    elif name == 'mae_moe_large':
        model = MAEMoE(in_channels, num_classes, embed_dim=1024, depth=24, num_heads=16, num_experts=16, top_k=4)
    else:
        raise ValueError(f'Unknown model: {name}')
    return model


def create_optimizers(name, model, max_epochs):
    if name == 'mininet':
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, max_epochs, 0.9)
    elif name == 'mininet_multimodal':
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, max_epochs, 0.9)
    elif name == 'segformer_b0':
        optimizer = torch.optim.AdamW(model.model.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, max_epochs, 0.1)
    elif name == 'segformer_b0_multimodal':
        optimizer = torch.optim.AdamW(model.model.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, max_epochs, 0.1)
    elif name == 'cmx_b0':
        optimizer = torch.optim.AdamW(model.model.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, max_epochs, 0.9)
    elif name == 'fusemoe':
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, max_epochs, 0.9)
    elif name in ['mae_moe', 'mae_moe_small', 'mae_moe_large']:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
        lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, max_epochs, 0.9)
    else:
        raise ValueError(f'Unknown model: {name}')

    return optimizer, lr_scheduler

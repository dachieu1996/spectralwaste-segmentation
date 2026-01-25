import torch

from .config import MoEConfig
from .sparse_moe import MoE


class FuseMoESegmentation(torch.nn.Module):
    """Segmentation wrapper that applies FuseMoE per pixel."""

    def __init__(
        self,
        in_channels: int | list[int],
        num_classes: int,
        embed_dim: int = 64,
        num_experts: int = 8,
        top_k: int = 2,
        noisy_gating: bool = False,
    ) -> None:
        super().__init__()

        if isinstance(in_channels, list):
            total_in_channels = sum(in_channels)
        else:
            total_in_channels = in_channels

        self.input_proj = torch.nn.Conv2d(total_in_channels, embed_dim, kernel_size=1)

        moe_config = MoEConfig(
            num_experts=num_experts,
            moe_input_size=embed_dim,
            moe_hidden_size=embed_dim * 4,
            moe_output_size=num_classes,
            router_type="joint",
            gating="softmax",
            num_modalities=1,
            top_k=top_k,
            noisy_gating=noisy_gating,
            dropout=0.1,
        )
        self.moe = MoE(moe_config)
        self.last_aux_loss: torch.Tensor | None = None

    def forward(self, inputs: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
        if isinstance(inputs, list):
            x = torch.concat(inputs, dim=1)
        else:
            x = inputs

        x = self.input_proj(x)
        batch_size, channels, height, width = x.shape

        tokens = x.permute(0, 2, 3, 1).reshape(batch_size * height * width, channels)
        logits_tokens, aux_loss = self.moe(tokens)
        self.last_aux_loss = aux_loss  # EDIT: store aux loss without changing train loop signature

        logits = logits_tokens.view(batch_size, height, width, -1).permute(0, 3, 1, 2).contiguous()
        return logits

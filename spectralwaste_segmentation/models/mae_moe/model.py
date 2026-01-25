# MAE-MoE: Self-Supervised Pretrained MoE for RGB+HSI Segmentation
import torch
import torch.nn as nn
import torch.nn.functional as F

from .vit_encoder import VisionTransformerEncoder
from .mae_encoder import DualBranchMAEEncoder
from .sparse_moe import MoE
from .config import MoEConfig


class MAEMoE(nn.Module):
    """MAE-MoE: Combines MAE pretrained encoders with MoE fusion
    
    Architecture:
    1. RGB Encoder: ViT encoder for RGB images
    2. HSI Encoder: Dual-branch MAE encoder for HSI data
    3. MoE Fusion: Per-modality MoE router for adaptive fusion
    4. Segmentation Head: Upsampling + Conv layers
    
    Based on:
    - FuseMoE: Mixture of Experts for multimodal fusion
    - SS-MAE: Self-supervised MAE for HSI data
    """
    
    def __init__(
        self,
        # Image settings
        img_size=224,
        patch_size=16,
        rgb_channels=3,
        hsi_channels=150,
        num_classes=10,
        
        # Encoder settings
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.,
        qkv_bias=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.,
        
        # MoE settings
        num_experts=8,
        top_k=4,
        router_type='permod',  # 'joint', 'permod', or 'disjoint'
        gating='softmax',  # 'softmax', 'laplace', or 'gaussian'
        noisy_gating=True,
        moe_hidden_size=2048,
        
        # Other
        use_cls_token=False,  # Edit: for segmentation, we don't use cls token
    ):
        """
        Args:
            img_size (int): input image size
            patch_size (int): patch size
            rgb_channels (int): number of RGB channels (default: 3)
            hsi_channels (int): number of HSI channels/bands
            num_classes (int): number of segmentation classes
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (float): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            num_experts (int): number of experts in MoE
            top_k (int): number of experts to select
            router_type (str): 'joint', 'permod', or 'disjoint'
            gating (str): gating function type
            noisy_gating (bool): whether to use noisy gating
            moe_hidden_size (int): hidden size of MoE experts
            use_cls_token (bool): whether to use cls token (False for segmentation)
        """
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.router_type = router_type
        
        # Calculate number of patches
        self.num_patches = (img_size // patch_size) ** 2
        
        # RGB Encoder
        self.rgb_encoder = VisionTransformerEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=rgb_channels,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            use_cls_token=use_cls_token,
        )
        
        # HSI Encoder (Dual-branch MAE)
        self.hsi_encoder = DualBranchMAEEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=hsi_channels,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
        )
        
        # MoE Fusion
        # Edit: Calculate MoE input/output sizes based on router type
        if router_type == 'joint':
            # Joint router: concatenate all features
            moe_input_size = embed_dim * 3  # RGB + HSI_spa + HSI_chan
            num_modalities = 1
        elif router_type == 'permod':
            # Per-modality router: separate routing per modality, shared experts
            moe_input_size = embed_dim * 3  # total size
            num_modalities = 3  # RGB, HSI_spa, HSI_chan
        else:  # disjoint
            # Disjoint router: separate routing and experts per modality
            moe_input_size = embed_dim * 3
            num_modalities = 3
        
        self.moe_config = MoEConfig(
            num_experts=num_experts,
            moe_input_size=moe_input_size,
            moe_hidden_size=moe_hidden_size,
            moe_output_size=embed_dim,
            router_type=router_type,
            gating=gating,
            num_modalities=num_modalities,
            top_k=top_k,
            noisy_gating=noisy_gating,
            dropout=drop_rate,
        )
        
        self.moe = MoE(self.moe_config)

        # Segmentation Head
        # Edit: Upsample from patch-level to pixel-level and predict classes
        self.seg_head = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=patch_size, mode='bilinear', align_corners=False),
            nn.Conv2d(embed_dim // 2, num_classes, kernel_size=1),
        )

        # Edit: Store aux_loss for load balancing
        self.aux_loss = None

    def forward(self, rgb, hsi, train=True):
        """
        Args:
            rgb: RGB image tensor of shape [B, 3, H, W]
            hsi: HSI image tensor of shape [B, C_hsi, H, W]
            train: whether in training mode (for noisy gating)

        Returns:
            output: segmentation output of shape [B, num_classes, H, W]
            aux_loss: MoE load balancing loss (only during training)
        """
        B, _, H, W = rgb.shape

        # Encode RGB
        rgb_features = self.rgb_encoder(rgb)  # [B, N, D] or [B, N+1, D] if cls_token

        # Encode HSI (dual-branch)
        hsi_features_spa, hsi_features_chan = self.hsi_encoder(hsi)  # [B, N+1, D], [B, N+1, D]

        # Edit: Remove cls tokens if present (for segmentation we need all patch features)
        if self.rgb_encoder.use_cls_token:
            rgb_features = rgb_features[:, 1:, :]  # Remove cls token
        hsi_features_spa = hsi_features_spa[:, 1:, :]  # Remove cls token
        hsi_features_chan = hsi_features_chan[:, 1:, :]  # Remove cls token

        # Edit: Channel branch has shape [B, C, D] not [B, N, D]
        # We need to expand it to match spatial dimensions
        # Repeat channel features for each spatial patch
        C = hsi_features_chan.shape[1]  # Number of channels
        N = self.num_patches  # Number of spatial patches
        # Expand: [B, C, D] -> [B, N, C, D] -> [B, N, C*D] or average over C
        # For simplicity, average over channels to get [B, D]
        hsi_chan_pooled = hsi_features_chan.mean(dim=1, keepdim=True)  # [B, 1, D]
        # Expand to all patches
        hsi_chan_expanded = hsi_chan_pooled.expand(B, N, self.embed_dim)  # [B, N, D]

        # Prepare features for MoE
        # Edit: Reshape to [B*N, D] for per-patch processing
        BN = B * N
        rgb_flat = rgb_features.reshape(BN, self.embed_dim)
        hsi_spa_flat = hsi_features_spa.reshape(BN, self.embed_dim)
        hsi_chan_flat = hsi_chan_expanded.reshape(BN, self.embed_dim)

        # MoE Fusion
        if self.router_type == 'joint':
            # Concatenate all modalities
            moe_input = torch.cat([rgb_flat, hsi_spa_flat, hsi_chan_flat], dim=1)
            fused_features, aux_loss = self.moe(moe_input, train=train)
        else:  # permod or disjoint
            # List of modality features
            moe_input = [rgb_flat, hsi_spa_flat, hsi_chan_flat]
            fused_features, aux_loss = self.moe(moe_input, train=train)

        # Edit: Store aux_loss for external access
        self.aux_loss = aux_loss

        # Reshape to spatial format
        h = w = int(self.num_patches ** 0.5)
        fused_features = fused_features.reshape(B, h, w, self.embed_dim)
        fused_features = fused_features.permute(0, 3, 1, 2)  # [B, D, h, w]

        # Segmentation head
        output = self.seg_head(fused_features)  # [B, num_classes, H, W]

        if train:
            return output, aux_loss
        else:
            return output


# Copy from base_models/SS-MAE/net/VIT/mae.py
# MAE model for pretraining RGB and HSI encoders

import torch
import torch.nn as nn
import torch.nn.functional as F
from .vit_encoder import VisionTransformerEncoder
from .mae_encoder import DualBranchMAEEncoder


class MaskTransLayerNorm(nn.Module):
    """Copy from SS-MAE: Normalization for each patch"""
    def __init__(self, hidden_size, eps=1e-12):
        super(MaskTransLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps
       
    def forward(self, x):
        u = x[:, :].mean(-1, keepdim=True)
        s = (x[:, :] - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta


class RGBMAEPretrainer(nn.Module):
    """MAE Pretrainer for RGB encoder
    
    Simplified from SS-MAE for RGB-only pretraining
    """
    def __init__(self, 
                 img_size=256,
                 patch_size=16,
                 in_chans=3,
                 encoder_dim=768,
                 encoder_depth=12,
                 encoder_heads=12,
                 decoder_dim=512,
                 decoder_depth=4,
                 decoder_heads=8,
                 mask_ratio=0.75):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.mask_ratio = mask_ratio
        
        # Encoder
        self.encoder = VisionTransformerEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=encoder_dim,
            depth=encoder_depth,
            num_heads=encoder_heads,
            use_cls_token=True
        )
        
        # Decoder: simple MLP decoder
        self.decoder_embed = nn.Linear(encoder_dim, decoder_dim)
        self.decoder_blocks = nn.Sequential(*[
            nn.TransformerEncoderLayer(
                d_model=decoder_dim,
                nhead=decoder_heads,
                dim_feedforward=decoder_dim * 4,
                batch_first=True
            ) for _ in range(decoder_depth)
        ])
        
        # Reconstruction head
        output_dim = in_chans * patch_size * patch_size
        self.decoder_pred = nn.Linear(decoder_dim, output_dim)
        self.patch_norm = MaskTransLayerNorm(output_dim)
        
    def random_masking(self, x, mask_ratio):
        """Copy from SS-MAE: Random masking
        
        Args:
            x: [B, N, D]
            mask_ratio: ratio of patches to mask
            
        Returns:
            x_masked: [B, N*(1-mask_ratio), D]
            mask: [B, N], 0 is keep, 1 is remove
            ids_restore: [B, N]
        """
        B, N, D = x.shape
        len_keep = int(N * (1 - mask_ratio))
        
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # Keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        # Generate binary mask: 0 is keep, 1 is remove
        mask = torch.ones([B, N], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore
    
    def forward(self, x):
        """Forward pass with masking
        
        Args:
            x: [B, C, H, W]
            
        Returns:
            pred: [B, C, H, W] reconstructed image
            mask: [B, N] binary mask
        """
        B = x.shape[0]
        
        # Encode
        features = self.encoder(x)  # [B, N+1, D]
        cls_token = features[:, :1, :]
        patch_features = features[:, 1:, :]  # [B, N, D]
        
        # Masking
        patch_features_masked, mask, ids_restore = self.random_masking(patch_features, self.mask_ratio)
        
        # Decode
        decoder_input = self.decoder_embed(patch_features_masked)  # [B, N_visible, decoder_dim]
        decoder_output = self.decoder_blocks(decoder_input)  # [B, N_visible, decoder_dim]
        
        # Predict
        pred_patches = self.decoder_pred(decoder_output)  # [B, N_visible, patch_size^2 * C]
        pred_patches = self.patch_norm(pred_patches)
        
        # EDIT: For simplicity, return only visible patches
        # In full MAE, you would restore all patches using ids_restore
        return pred_patches, mask, ids_restore, patch_features_masked


class HSIMAEPretrainer(nn.Module):
    """MAE Pretrainer for HSI encoder with dual-branch

    Copy from SS-MAE architecture
    """
    def __init__(self,
                 img_size=256,
                 patch_size=16,
                 in_chans=150,
                 encoder_dim=768,
                 encoder_depth=12,
                 encoder_heads=12,
                 decoder_dim=512,
                 decoder_depth=4,
                 decoder_heads=8,
                 mask_ratio=0.75):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches_spa = (img_size // patch_size) ** 2
        self.num_patches_chan = in_chans
        self.mask_ratio = mask_ratio
        self.in_chans = in_chans

        # Dual-branch encoder
        self.encoder = DualBranchMAEEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=encoder_dim,
            depth=encoder_depth,
            num_heads=encoder_heads
        )

        # Projection layers (copy from SS-MAE)
        self.proj_spa = nn.Linear(encoder_dim, decoder_dim)
        self.proj_chan = nn.Linear(encoder_dim, decoder_dim)

        # Decoder blocks for spatial branch
        self.decoder_spa = nn.Sequential(*[
            nn.TransformerEncoderLayer(
                d_model=decoder_dim,
                nhead=decoder_heads,
                dim_feedforward=decoder_dim * 4,
                batch_first=True
            ) for _ in range(decoder_depth)
        ])

        # Decoder blocks for channel branch
        self.decoder_chan = nn.Sequential(*[
            nn.TransformerEncoderLayer(
                d_model=decoder_dim,
                nhead=decoder_heads,
                dim_feedforward=decoder_dim * 4,
                batch_first=True
            ) for _ in range(decoder_depth)
        ])

        # Reconstruction heads (copy from SS-MAE)
        output_dim_spa = in_chans  # Reconstruct channels
        output_dim_chan = img_size * img_size  # Reconstruct spatial
        self.restruction_spa = nn.Linear(decoder_dim, output_dim_spa)
        self.restruction_chan = nn.Linear(decoder_dim, output_dim_chan)
        self.patch_norm_spa = MaskTransLayerNorm(output_dim_spa)
        self.patch_norm_chan = MaskTransLayerNorm(output_dim_chan)

        # Restore image (copy from SS-MAE)
        self.unconv_spa = nn.ConvTranspose2d(output_dim_spa, in_chans, patch_size, patch_size)

    def random_masking(self, x, mask_ratio):
        """Copy from SS-MAE: Random masking"""
        B, N, D = x.shape
        len_keep = int(N * (1 - mask_ratio))

        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        mask = torch.ones([B, N], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward(self, x):
        """Forward pass with dual-branch masking

        Args:
            x: [B, C, H, W]

        Returns:
            restore_image_spa: [B, C, H, W]
            mask_spa: [B, N_spa]
            restore_image_chan: [B, C, H, W]
            mask_chan: [B, N_chan]
        """
        B, C, H, W = x.shape

        # Encode with dual-branch
        spa_features, chan_features = self.encoder(x)  # [B, N_spa+1, D], [B, N_chan+1, D]

        # Remove cls tokens
        spa_patches = spa_features[:, 1:, :]  # [B, N_spa, D]
        chan_patches = chan_features[:, 1:, :]  # [B, N_chan, D]

        # Masking
        spa_masked, mask_spa, ids_restore_spa = self.random_masking(spa_patches, self.mask_ratio)
        chan_masked, mask_chan, ids_restore_chan = self.random_masking(chan_patches, self.mask_ratio)

        # Project to decoder dim
        spa_decoder_input = self.proj_spa(spa_masked)
        chan_decoder_input = self.proj_chan(chan_masked)

        # Decode
        spa_decoded = self.decoder_spa(spa_decoder_input)
        chan_decoded = self.decoder_chan(chan_decoder_input)

        # Reconstruct
        outputs_spa = self.restruction_spa(spa_decoded)  # [B, N_visible, C]
        outputs_chan = self.restruction_chan(chan_decoded)  # [B, C_visible, H*W]

        # Normalize
        outputs_spa = self.patch_norm_spa(outputs_spa)
        outputs_chan = self.patch_norm_chan(outputs_chan)

        # EDIT: Return outputs and masks for loss computation
        return outputs_spa, mask_spa, outputs_chan, mask_chan



# MAE Encoder for RGB and HSI modalities
# Based on base_models/SS-MAE/net/VIT/mae.py
import torch
import torch.nn as nn
from functools import partial

from .vit_encoder import VisionTransformerEncoder, Block, Block_wo_gate
from .patch_embed import PatchEmbed_spa, PatchEmbed_chan, PositionEmbed
from .vit_layers import trunc_normal_


class DualBranchMAEEncoder(nn.Module):
    """Dual-branch MAE Encoder for HSI data
    
    Processes HSI data with two branches:
    - Spatial branch: treats channels as batch dimension
    - Channel branch: treats spatial dimensions as features
    
    Based on SS-MAE architecture.
    """
    
    def __init__(self, img_size=224, patch_size=16, in_chans=150, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0., 
                 drop_path_rate=0., norm_layer=None, act_layer=None):
        """
        Args:
            img_size (int): input image size
            patch_size (int): patch size
            in_chans (int): number of input channels (HSI bands)
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
            act_layer: (nn.Module): activation layer
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.in_chans = in_chans
        
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        
        # Spatial branch: patch embedding for spatial dimension
        self.patch_embed_spa = PatchEmbed_spa(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches_spa = self.patch_embed_spa.num_patches
        
        # Channel branch: patch embedding for channel dimension
        self.patch_embed_chan = PatchEmbed_chan(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches_chan = self.patch_embed_chan.num_patches
        
        # CLS tokens
        self.cls_token_spa = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_token_chan = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Position embeddings
        self.pos_embed_spa = nn.Parameter(torch.zeros(1, num_patches_spa + 1, embed_dim))
        self.pos_embed_chan = nn.Parameter(torch.zeros(1, num_patches_chan + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        # Spatial branch blocks
        self.blocks_spa = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        
        # Channel branch blocks
        self.blocks_chan = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        
        self.norm = norm_layer(embed_dim)
        
        self.apply(self._init_weights)
        trunc_normal_(self.pos_embed_spa, std=.02)
        trunc_normal_(self.pos_embed_chan, std=.02)
        trunc_normal_(self.cls_token_spa, std=.02)
        trunc_normal_(self.cls_token_chan, std=.02)

    def _init_weights(self, module):
        """ Weight initialization
        """
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        """
        Args:
            x: input tensor of shape [B, C, H, W] where C is number of HSI bands
        Returns:
            features_spa: spatial branch features [B, N_spa+1, D]
            features_chan: channel branch features [B, N_chan+1, D]
        """
        B = x.shape[0]
        
        # Spatial branch
        x_spa = self.patch_embed_spa(x)  # [B, N_spa, D]
        cls_tokens_spa = self.cls_token_spa.expand(B, -1, -1)  # [B, 1, D]
        x_spa = torch.cat((cls_tokens_spa, x_spa), dim=1)  # [B, N_spa+1, D]
        x_spa = x_spa + self.pos_embed_spa
        x_spa = self.pos_drop(x_spa)
        x_spa = self.blocks_spa(x_spa)
        x_spa = self.norm(x_spa)
        
        # Channel branch
        x_chan = self.patch_embed_chan(x)  # [B, N_chan, D]
        cls_tokens_chan = self.cls_token_chan.expand(B, -1, -1)  # [B, 1, D]
        x_chan = torch.cat((cls_tokens_chan, x_chan), dim=1)  # [B, N_chan+1, D]
        x_chan = x_chan + self.pos_embed_chan
        x_chan = self.pos_drop(x_chan)
        x_chan = self.blocks_chan(x_chan)
        x_chan = self.norm(x_chan)
        
        return x_spa, x_chan


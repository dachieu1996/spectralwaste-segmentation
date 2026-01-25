# Copy from base_models/SS-MAE/net/VIT/layers/patch_embd.py
import torch
import torch.nn as nn
import math


class PatchEmbed_spa(nn.Module):
    """ 2D Image to Patch Embedding for spatial dimension
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class PatchEmbed_chan(nn.Module):
    """ 2D Image to Patch Embedding for channel dimension
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = in_chans
        self.flatten = flatten

        self.proj = nn.Linear(img_size[0] * img_size[1], embed_dim)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = x.reshape(B, C, -1)  # B, C, H*W
        x = self.proj(x)  # B, C, embed_dim
        # Edit: Keep shape as [B, C, embed_dim] for channel patches
        x = self.norm(x)
        return x


class PositionEmbed(nn.Module):
    """ Sinusoidal Position Embedding
    """
    def __init__(self, length, embed_dim, args=None):
        super().__init__()
        self.length = length
        self.embed_dim = embed_dim
        if args is not None:
            self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create sinusoidal position embeddings
        position = torch.arange(0, length).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim))
        
        pos_embed = torch.zeros(1, length, embed_dim)
        pos_embed[0, :, 0::2] = torch.sin(position * div_term)
        pos_embed[0, :, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pos_embed', pos_embed)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [B, N, D] where N <= self.length
        Returns:
            Tensor of shape [B, N, D] with position embeddings added
        """
        B, N, D = x.shape
        return x + self.pos_embed[:, :N, :].to(x.device)


# Edit: Add helper function for creating position embeddings
def get_sinusoid_encoding_table(n_position, d_hid):
    """Sinusoid position encoding table"""
    def get_position_angle_vec(position):
        return [position / torch.pow(torch.tensor(10000.0), 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = torch.FloatTensor([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = torch.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = torch.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return sinusoid_table.unsqueeze(0)


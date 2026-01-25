# Copy from base_models/SS-MAE/loss/mae_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F 
from einops import rearrange


def build_mask_spa(mask_index, patch_size, img_size):
    """Build spatial mask for MAE pretraining
    
    Args:
        mask_index: List of patch indices to mask (1-indexed)
        patch_size: Size of each patch
        img_size: Size of input image
        
    Returns:
        mask_map: Binary mask of shape (img_size, img_size)
    """
    num_pathces = img_size // patch_size
    mask_map = torch.zeros((img_size, img_size)).float()
    # reshape the h w -> n c 
    mask_map = rearrange(mask_map, '(h p1) (w p2) -> (h w) (p1 p2)', h=num_pathces, w=num_pathces, p1=patch_size, p2=patch_size)
    mask_index = [index-1 for index in mask_index ]
    mask_map[mask_index] = 1.
    # reshape the n c -> h w
    mask_map = rearrange(mask_map, '(h w) (p1 p2) -> (h p1) (w p2)', h=num_pathces, w=num_pathces, p1=patch_size, p2=patch_size)
    return mask_map


def build_mask_chan(mask_index, channel_num, patch_size):
    """Build channel mask for MAE pretraining
    
    Args:
        mask_index: List of channel indices to mask (1-indexed)
        channel_num: Number of channels
        patch_size: Size of each patch (not used but kept for compatibility)
        
    Returns:
        mask_map: Binary mask of shape (channel_num, 1)
    """
    mask_map = torch.zeros((channel_num, 1)).float()
    mask_index = [index-1 for index in mask_index ]
    mask_map[mask_index] = 1.
    return mask_map


class MSELoss(nn.Module):
    """MSE Loss for MAE reconstruction"""
    
    def __init__(self, device):
        super().__init__()
        self.device = device
        
    def forward(self, pred, target, mask_map):
        """Compute MSE loss on masked regions
        
        Args:
            pred: Predicted reconstruction
            target: Ground truth
            mask_map: Binary mask indicating masked regions
            
        Returns:
            loss: MSE loss
        """
        pred = pred * mask_map.to(self.device)
        target = target * mask_map.to(self.device)
        loss = F.mse_loss(pred, target)
        return loss


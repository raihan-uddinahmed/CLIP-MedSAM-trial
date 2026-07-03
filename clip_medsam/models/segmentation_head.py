"""
Segmentation Head module for combining vision and language features.

This module takes features from both vision and language encoders
and produces segmentation masks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SegmentationHead(nn.Module):
    """
    Segmentation head that combines visual and language features.
    
    Uses cross-attention and fusion to create segmentation masks
    guided by both image content and text prompts.
    
    Args:
        vision_dim (int): Dimension of vision features
        language_dim (int): Dimension of language features
        output_dim (int): Number of segmentation classes
        hidden_dim (int): Hidden dimension for intermediate layers
    """
    
    def __init__(self, vision_dim=2048, language_dim=512, 
                 output_dim=1, hidden_dim=256):
        super(SegmentationHead, self).__init__()
        
        self.vision_dim = vision_dim
        self.language_dim = language_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        
        # Vision projection
        self.vision_proj = nn.Sequential(
            nn.Conv2d(vision_dim, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # Language projection and expansion
        self.language_proj = nn.Sequential(
            nn.Linear(language_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Fusion layers
        self.fusion = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # Output segmentation head
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, output_dim, kernel_size=1)
        )
    
    def forward(self, vision_features, language_features):
        """
        Generate segmentation masks from vision and language features.
        
        Args:
            vision_features (torch.Tensor): Visual features (B, vision_dim, H, W)
            language_features (torch.Tensor): Language features (B, language_dim)
        
        Returns:
            torch.Tensor: Segmentation masks (B, output_dim, H, W)
        """
        batch_size, _, height, width = vision_features.shape
        
        # Project vision features
        vision_proj = self.vision_proj(vision_features)  # (B, hidden_dim, H, W)
        
        # Project and expand language features
        language_proj = self.language_proj(language_features)  # (B, hidden_dim)
        language_expanded = language_proj.unsqueeze(-1).unsqueeze(-1)  # (B, hidden_dim, 1, 1)
        language_expanded = language_expanded.expand(-1, -1, height, width)  # (B, hidden_dim, H, W)
        
        # Fuse features via concatenation
        fused = torch.cat([vision_proj, language_expanded], dim=1)  # (B, hidden_dim*2, H, W)
        
        # Apply fusion layers
        fused = self.fusion(fused)
        
        # Generate segmentation masks
        masks = self.segmentation_head(fused)
        
        return masks


class CrossAttentionSegmentationHead(nn.Module):
    """
    Segmentation head using cross-attention mechanism.
    
    Employs multi-head cross-attention to align visual and language features,
    providing more sophisticated feature fusion.
    
    Args:
        vision_dim (int): Dimension of vision features
        language_dim (int): Dimension of language features
        output_dim (int): Number of segmentation classes
        n_heads (int): Number of attention heads
        hidden_dim (int): Hidden dimension
    """
    
    def __init__(self, vision_dim=2048, language_dim=512, 
                 output_dim=1, n_heads=8, hidden_dim=256):
        super(CrossAttentionSegmentationHead, self).__init__()
        
        self.vision_dim = vision_dim
        self.language_dim = language_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.hidden_dim = hidden_dim
        
        # Vision projection
        self.vision_proj = nn.Conv2d(vision_dim, hidden_dim, kernel_size=1)
        
        # Language projection
        self.language_proj = nn.Linear(language_dim, hidden_dim)
        
        # Cross-attention layers
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            batch_first=True
        )
        
        # Fusion and segmentation head
        self.fusion = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, output_dim, kernel_size=1)
        )
    
    def forward(self, vision_features, language_features):
        """
        Generate segmentation masks using cross-attention.
        
        Args:
            vision_features (torch.Tensor): Visual features (B, vision_dim, H, W)
            language_features (torch.Tensor): Language features (B, language_dim)
        
        Returns:
            torch.Tensor: Segmentation masks (B, output_dim, H, W)
        """
        batch_size, _, height, width = vision_features.shape
        
        # Project features
        vision_proj = self.vision_proj(vision_features)  # (B, hidden_dim, H, W)
        language_proj = self.language_proj(language_features)  # (B, hidden_dim)
        
        # Reshape vision features for attention
        vision_flat = vision_proj.flatten(2).transpose(1, 2)  # (B, H*W, hidden_dim)
        language_query = language_proj.unsqueeze(1)  # (B, 1, hidden_dim)
        
        # Apply cross-attention
        attended, _ = self.cross_attention(
            query=language_query,
            key=vision_flat,
            value=vision_flat
        )  # (B, 1, hidden_dim)
        
        # Broadcast attention output and add to vision features
        attended_expanded = attended.squeeze(1).unsqueeze(-1).unsqueeze(-1)  # (B, hidden_dim, 1, 1)
        attended_expanded = attended_expanded.expand_as(vision_proj)
        
        fused = vision_proj + attended_expanded
        
        # Generate segmentation masks
        masks = self.fusion(fused)
        
        return masks


class MultiScaleSegmentationHead(nn.Module):
    """
    Multi-scale segmentation head processing features at multiple scales.
    
    Combines coarse and fine features for better segmentation accuracy.
    """
    
    def __init__(self, vision_dims=[256, 512, 1024, 2048], 
                 language_dim=512, output_dim=1, hidden_dim=256):
        super(MultiScaleSegmentationHead, self).__init__()
        
        self.vision_dims = vision_dims
        self.language_dim = language_dim
        self.output_dim = output_dim
        
        # Create heads for each scale
        self.heads = nn.ModuleList([
            SegmentationHead(v_dim, language_dim, output_dim, hidden_dim)
            for v_dim in vision_dims
        ])
        
        # Fusion of multi-scale outputs
        self.fusion = nn.Sequential(
            nn.Conv2d(output_dim * len(vision_dims), hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, output_dim, kernel_size=1)
        )
    
    def forward(self, vision_features_list, language_features):
        """
        Generate segmentation masks from multi-scale features.
        
        Args:
            vision_features_list (list): List of visual features at different scales
            language_features (torch.Tensor): Language features (B, language_dim)
        
        Returns:
            torch.Tensor: Segmentation masks (B, output_dim, H, W)
        """
        # Get target size from first feature map
        target_size = vision_features_list[0].shape[-2:]
        
        # Generate masks at each scale
        masks_list = []
        for i, (head, features) in enumerate(zip(self.heads, vision_features_list)):
            masks = head(features, language_features)
            # Resize to target size
            masks = F.interpolate(masks, size=target_size, mode='bilinear', align_corners=False)
            masks_list.append(masks)
        
        # Concatenate and fuse
        fused = torch.cat(masks_list, dim=1)
        output_masks = self.fusion(fused)
        
        return output_masks

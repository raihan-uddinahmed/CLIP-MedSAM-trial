"""
Vision Encoder module using Segment Anything Model (SAM) architecture.

This module provides the vision encoder that extracts image features
using ViT-based architecture similar to SAM.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class VisionEncoder(nn.Module):
    """
    Vision Encoder based on Vision Transformers (ViT).
    
    Extracts high-level visual features from input images.
    Designed to be compatible with Segment Anything Model (SAM) architecture.
    
    Args:
        model_type (str): Type of vision model ('vit_base', 'vit_large', 'resnet50')
        pretrained (bool): Whether to use pretrained weights
        freeze (bool): Whether to freeze encoder weights during training
    """
    
    def __init__(self, model_type='vit_base', pretrained=True, freeze=False):
        super(VisionEncoder, self).__init__()
        
        self.model_type = model_type
        self.pretrained = pretrained
        self.freeze = freeze
        
        # Initialize encoder based on model type
        if model_type == 'resnet50':
            self.encoder = models.resnet50(pretrained=pretrained)
            # Remove classification layer
            self.encoder = nn.Sequential(*list(self.encoder.children())[:-1])
            self.feature_dim = 2048
        
        elif model_type == 'vit_base':
            try:
                from torchvision.models import vision_transformer
                self.encoder = vision_transformer.vit_b_16(pretrained=pretrained)
                self.feature_dim = 768
            except:
                # Fallback to ResNet if ViT not available
                self.encoder = models.resnet50(pretrained=pretrained)
                self.encoder = nn.Sequential(*list(self.encoder.children())[:-1])
                self.feature_dim = 2048
        
        elif model_type == 'vit_large':
            try:
                from torchvision.models import vision_transformer
                self.encoder = vision_transformer.vit_l_16(pretrained=pretrained)
                self.feature_dim = 1024
            except:
                # Fallback to ResNet if ViT not available
                self.encoder = models.resnet50(pretrained=pretrained)
                self.encoder = nn.Sequential(*list(self.encoder.children())[:-1])
                self.feature_dim = 2048
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Freeze encoder if requested
        if freeze:
            self._freeze_encoder()
    
    def _freeze_encoder(self):
        """Freeze encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        """
        Extract visual features from input image.
        
        Args:
            x (torch.Tensor): Input image tensor of shape (B, 3, H, W)
        
        Returns:
            torch.Tensor: Visual features of shape (B, feature_dim, H', W')
                         or (B, n_patches, feature_dim) for ViT
        """
        features = self.encoder(x)
        return features
    
    def get_feature_dim(self):
        """Get feature dimension of the encoder."""
        return self.feature_dim


class MultiScaleVisionEncoder(nn.Module):
    """
    Multi-scale Vision Encoder extracting features at multiple scales.
    
    Useful for capturing both fine-grained and coarse features.
    """
    
    def __init__(self, model_type='resnet50', pretrained=True):
        super(MultiScaleVisionEncoder, self).__init__()
        
        if model_type == 'resnet50':
            backbone = models.resnet50(pretrained=pretrained)
            
            # Extract different scales
            self.layer1 = nn.Sequential(*list(backbone.children())[:5])  # 256
            self.layer2 = nn.Sequential(*list(backbone.children())[:6])  # 512
            self.layer3 = nn.Sequential(*list(backbone.children())[:7])  # 1024
            self.layer4 = nn.Sequential(*list(backbone.children())[:8])  # 2048
            
            self.feature_dims = [256, 512, 1024, 2048]
        
        else:
            raise ValueError(f"Multi-scale not implemented for {model_type}")
    
    def forward(self, x):
        """
        Extract multi-scale features.
        
        Args:
            x (torch.Tensor): Input image tensor
        
        Returns:
            list: List of feature tensors at different scales
        """
        f1 = self.layer1(x)
        f2 = self.layer2(x)
        f3 = self.layer3(x)
        f4 = self.layer4(x)
        
        return [f1, f2, f3, f4]
    
    def get_feature_dims(self):
        """Get feature dimensions at each scale."""
        return self.feature_dims

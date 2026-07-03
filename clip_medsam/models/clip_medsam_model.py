"""
Main CLIP-MedSAM model combining vision and language encoders with segmentation head.

This is the primary model class that users interact with for segmentation tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .vision_encoder import VisionEncoder, MultiScaleVisionEncoder
from .language_encoder import LanguageEncoder, MedicalTermEncoder
from .segmentation_head import SegmentationHead, CrossAttentionSegmentationHead


class CLIPMedSAM(nn.Module):
    """
    CLIP-MedSAM: Vision Language Model for Medical Image Segmentation.
    
    Combines CLIP for text understanding and vision encoding with a segmentation head
    for precise medical image segmentation guided by natural language prompts.
    
    Args:
        vision_model_type (str): Type of vision encoder ('vit_base', 'vit_large', 'resnet50')
        language_model_name (str): CLIP model to use
        use_medical_encoder (bool): Whether to use specialized medical term encoder
        use_cross_attention (bool): Whether to use cross-attention in segmentation head
        use_multi_scale (bool): Whether to use multi-scale features
        num_classes (int): Number of segmentation classes
        device (str): Device to use ('cpu' or 'cuda')
    """
    
    def __init__(
        self,
        vision_model_type='resnet50',
        language_model_name='openai/clip-vit-base-patch32',
        use_medical_encoder=True,
        use_cross_attention=False,
        use_multi_scale=False,
        num_classes=1,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super(CLIPMedSAM, self).__init__()
        
        self.vision_model_type = vision_model_type
        self.language_model_name = language_model_name
        self.use_medical_encoder = use_medical_encoder
        self.use_multi_scale = use_multi_scale
        self.num_classes = num_classes
        self.device_name = device
        
        # Initialize encoders
        if use_multi_scale:
            self.vision_encoder = MultiScaleVisionEncoder(
                model_type=vision_model_type,
                pretrained=True
            )
            vision_dim = self.vision_encoder.get_feature_dims()
        else:
            self.vision_encoder = VisionEncoder(
                model_type=vision_model_type,
                pretrained=True,
                freeze=False
            )
            vision_dim = self.vision_encoder.get_feature_dim()
        
        if use_medical_encoder:
            self.language_encoder = MedicalTermEncoder(
                model_name=language_model_name,
                freeze=True
            )
        else:
            self.language_encoder = LanguageEncoder(
                model_name=language_model_name,
                freeze=True
            )
        
        language_dim = self.language_encoder.get_feature_dim()
        
        # Initialize segmentation head
        if use_multi_scale:
            from .segmentation_head import MultiScaleSegmentationHead
            self.segmentation_head = MultiScaleSegmentationHead(
                vision_dims=vision_dim,
                language_dim=language_dim,
                output_dim=num_classes
            )
        elif use_cross_attention:
            self.segmentation_head = CrossAttentionSegmentationHead(
                vision_dim=vision_dim if not isinstance(vision_dim, list) else vision_dim[0],
                language_dim=language_dim,
                output_dim=num_classes
            )
        else:
            self.segmentation_head = SegmentationHead(
                vision_dim=vision_dim if not isinstance(vision_dim, list) else vision_dim[0],
                language_dim=language_dim,
                output_dim=num_classes
            )
        
        self.to(device)
    
    def forward(self, images, text_prompts):
        """
        Forward pass for segmentation.
        
        Args:
            images (torch.Tensor): Input images (B, 3, H, W)
            text_prompts (list): List of text prompts for segmentation
        
        Returns:
            torch.Tensor: Segmentation masks (B, num_classes, H, W)
        """
        # Extract visual features
        if self.use_multi_scale:
            vision_features = self.vision_encoder(images)
        else:
            vision_features = self.vision_encoder(images)
        
        # Extract language features
        language_features = self.language_encoder(text_prompts)
        
        # Generate segmentation masks
        if self.use_multi_scale:
            masks = self.segmentation_head(vision_features, language_features)
        else:
            masks = self.segmentation_head(vision_features, language_features)
        
        return masks
    
    def segment(self, image, text_prompts):
        """
        Segment an image based on text prompts.
        
        Args:
            image (np.ndarray or torch.Tensor): Input image
            text_prompts (list or str): Text prompt(s) for segmentation
        
        Returns:
            np.ndarray: Segmentation masks
        """
        self.eval()
        
        with torch.no_grad():
            # Convert image to tensor if needed
            if isinstance(image, np.ndarray):
                image_tensor = torch.from_numpy(image).float()
                if len(image_tensor.shape) == 2:  # Grayscale
                    image_tensor = image_tensor.unsqueeze(0).repeat(3, 1, 1)
                if len(image_tensor.shape) == 3:  # Add batch dimension
                    image_tensor = image_tensor.unsqueeze(0)
                # Normalize to [0, 1]
                if image_tensor.max() > 1.0:
                    image_tensor = image_tensor / 255.0
            else:
                image_tensor = image
            
            # Ensure 4D tensor (B, C, H, W)
            if len(image_tensor.shape) == 3:
                image_tensor = image_tensor.unsqueeze(0)
            
            # Normalize image
            image_tensor = self._normalize_image(image_tensor)
            image_tensor = image_tensor.to(self.device_name)
            
            # Ensure text prompts is a list
            if isinstance(text_prompts, str):
                text_prompts = [text_prompts]
            
            # Forward pass
            masks = self.forward(image_tensor, text_prompts)
            
            # Apply sigmoid for binary segmentation
            if self.num_classes == 1:
                masks = torch.sigmoid(masks)
            else:
                masks = torch.softmax(masks, dim=1)
        
        # Convert to numpy
        masks_np = masks.detach().cpu().numpy()
        
        return masks_np
    
    def batch_segment(self, images, text_prompts):
        """
        Segment multiple images.
        
        Args:
            images (list): List of input images
            text_prompts (list or str): Text prompt(s)
        
        Returns:
            np.ndarray: Batch of segmentation masks
        """
        all_masks = []
        
        for image in images:
            masks = self.segment(image, text_prompts)
            all_masks.append(masks)
        
        # Stack masks
        all_masks = np.concatenate(all_masks, axis=0)
        
        return all_masks
    
    def _normalize_image(self, image):
        """
        Normalize image to standard ImageNet statistics.
        
        Args:
            image (torch.Tensor): Input image tensor
        
        Returns:
            torch.Tensor: Normalized image
        """
        mean = torch.tensor([0.485, 0.456, 0.406]).to(self.device_name).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).to(self.device_name).view(1, 3, 1, 1)
        
        return (image - mean) / std
    
    def to(self, device):
        """Move model to device."""
        self.device_name = device
        return super().to(device)
    
    def save_model(self, path):
        """Save model checkpoint."""
        torch.save(self.state_dict(), path)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """Load model checkpoint."""
        self.load_state_dict(torch.load(path, map_location=self.device_name))
        print(f"Model loaded from {path}")

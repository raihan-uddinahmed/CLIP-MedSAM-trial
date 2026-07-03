"""
Language Encoder module using CLIP text encoder.

This module provides the language encoder that extracts semantic features
from text prompts using CLIP's text encoder.
"""

import torch
import torch.nn as nn
from transformers import CLIPTextModel, CLIPTokenizer


class LanguageEncoder(nn.Module):
    """
    Language Encoder based on CLIP text encoder.
    
    Converts text prompts into semantic embeddings that can be compared
    with visual features for segmentation.
    
    Args:
        model_name (str): CLIP model to use ('openai/clip-vit-base-patch32', etc.)
        freeze (bool): Whether to freeze encoder weights during training
    """
    
    def __init__(self, model_name='openai/clip-vit-base-patch32', freeze=True):
        super(LanguageEncoder, self).__init__()
        
        self.model_name = model_name
        
        try:
            self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
            self.encoder = CLIPTextModel.from_pretrained(model_name)
        except Exception as e:
            print(f"Warning: Could not load {model_name}. Attempting fallback...")
            self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
            self.encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        
        self.feature_dim = self.encoder.config.hidden_size
        
        if freeze:
            self._freeze_encoder()
    
    def _freeze_encoder(self):
        """Freeze encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def forward(self, texts):
        """
        Encode text prompts into embeddings.
        
        Args:
            texts (list or str): Text prompt(s) to encode
                                Can be a single string or list of strings
        
        Returns:
            torch.Tensor: Text embeddings of shape (batch_size, feature_dim)
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Tokenize text
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=77
        )
        
        # Move to same device as encoder
        device = next(self.encoder.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Extract embeddings
        outputs = self.encoder(**inputs)
        
        # Get pooled embeddings
        embeddings = outputs.pooler_output
        
        return embeddings
    
    def get_feature_dim(self):
        """Get feature dimension of the encoder."""
        return self.feature_dim
    
    def encode_single(self, text):
        """
        Encode a single text prompt.
        
        Args:
            text (str): Text to encode
        
        Returns:
            torch.Tensor: Embedding of shape (1, feature_dim)
        """
        return self.forward([text])
    
    def encode_batch(self, texts):
        """
        Encode multiple text prompts.
        
        Args:
            texts (list): List of texts to encode
        
        Returns:
            torch.Tensor: Embeddings of shape (len(texts), feature_dim)
        """
        return self.forward(texts)


class MedicalTermEncoder(nn.Module):
    """
    Specialized language encoder for medical terminology.
    
    Handles medical abbreviations and synonyms to better encode
    anatomical structures and pathological findings.
    """
    
    def __init__(self, model_name='openai/clip-vit-base-patch32', freeze=True):
        super(MedicalTermEncoder, self).__init__()
        
        self.base_encoder = LanguageEncoder(model_name=model_name, freeze=freeze)
        
        # Medical term synonyms and expansions
        self.medical_expansions = {
            'ct': 'computed tomography',
            'mri': 'magnetic resonance imaging',
            'us': 'ultrasound',
            'xray': 'x-ray',
            'lwc': 'left ventricle',
            'rwc': 'right ventricle',
            'ca': 'cancer',
            'mi': 'myocardial infarction',
            'stroke': 'cerebrovascular accident',
            'tumor': 'malignant neoplasm',
            'lesion': 'abnormal tissue',
        }
    
    def _expand_medical_terms(self, text):
        """
        Expand abbreviated medical terms.
        
        Args:
            text (str): Text containing medical terms
        
        Returns:
            str: Expanded text
        """
        expanded = text.lower()
        for abbrev, expansion in self.medical_expansions.items():
            expanded = expanded.replace(abbrev, expansion)
        return expanded
    
    def forward(self, texts):
        """
        Encode medical text prompts.
        
        Args:
            texts (list or str): Medical text(s) to encode
        
        Returns:
            torch.Tensor: Embeddings
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Expand medical terms
        expanded_texts = [self._expand_medical_terms(text) for text in texts]
        
        # Encode
        return self.base_encoder.forward(expanded_texts)
    
    def get_feature_dim(self):
        """Get feature dimension."""
        return self.base_encoder.get_feature_dim()

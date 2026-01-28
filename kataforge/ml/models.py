"""
KataForge - Adaptive Martial Arts Analysis System
Copyright © 2026 DeMoD LLC. All rights reserved.

This file is part of KataForge, released under the KataForge License
(based on Elastic License v2). See LICENSE in the project root for full terms.

SPDX-License-Identifier: Elastic-2.0

Description:
    [Brief module description – please edit]

Usage notes:
    - Private self-hosting, dojo use, and modifications are permitted.
    - Offering as a hosted/managed service to third parties is prohibited
      without explicit written permission from DeMoD LLC.
"""

"""Machine learning models for martial arts technique analysis."""

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import SAGEConv, global_mean_pool
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None
    SAGEConv = None
    global_mean_pool = None

import numpy as np
from typing import Dict, List, Optional, Tuple


class Scale(nn.Module):
    """Custom module to scale tensor by a constant factor.
    
    This is needed because lambda functions cannot be used in nn.Sequential.
    """
    def __init__(self, scale_factor: float):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for this module")
        super().__init__()
        self.scale_factor = scale_factor
    
    def forward(self, x):
        return x * self.scale_factor


class GraphSAGEModel(nn.Module):
    """Graph Neural Network for skeletal pose analysis.
    
    Uses GraphSAGE convolutions to learn representations of human poses
    by treating the skeleton as a graph where joints are nodes and bones
    are edges.
    
    Features:
    - Multi-layer GraphSAGE with residual connections
    - Batch normalization for training stability
    - Configurable depth and width
    """
    
    # MediaPipe skeleton connectivity (33 landmarks)
    SKELETON_EDGES = [
        # Face
        (0, 1), (1, 2), (2, 3), (3, 7),  # Right eye path
        (0, 4), (4, 5), (5, 6), (6, 8),  # Left eye path
        (0, 9), (0, 10),  # Mouth connections
        # Upper body
        (11, 12),  # Shoulder to shoulder
        (11, 13), (13, 15),  # Left arm
        (12, 14), (14, 16),  # Right arm
        (11, 23), (12, 24),  # Shoulders to hips
        # Hands (simplified)
        (15, 17), (15, 19), (15, 21),  # Left hand
        (16, 18), (16, 20), (16, 22),  # Right hand
        # Lower body
        (23, 24),  # Hip to hip
        (23, 25), (25, 27),  # Left leg
        (24, 26), (26, 28),  # Right leg
        # Feet
        (27, 29), (27, 31),  # Left foot
        (28, 30), (28, 32),  # Right foot
    ]
    
    def __init__(self, 
                 num_features: int = 4,  # x, y, z, visibility
                 hidden_channels: int = 128,
                 num_classes: int = 10,
                 num_layers: int = 3,
                 dropout: float = 0.3):
        """Initialize GraphSAGE model.
        
        Args:
            num_features: Number of features per node (landmark)
            hidden_channels: Hidden layer dimension
            num_classes: Number of classification classes
            num_layers: Number of GraphSAGE layers
            dropout: Dropout rate
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch and PyTorch Geometric are required for this model")
            
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Create edge index tensor
        self.register_buffer('edge_index', self._create_skeleton_graph())
        
        # Input projection
        self.input_proj = nn.Linear(num_features, hidden_channels)
        
        # GraphSAGE layers with batch norm
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for _ in range(num_layers):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.BatchNorm1d(hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Kaiming initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def _create_skeleton_graph(self) -> torch.Tensor:
        """Create edge index tensor for human skeleton connectivity."""
        # Make edges bidirectional
        edges = []
        for src, dst in self.SKELETON_EDGES:
            edges.append((src, dst))
            edges.append((dst, src))
        
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        return edge_index
    
    def forward(self, x: torch.Tensor, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Node features [num_nodes, num_features] or [batch, 33, num_features]
            batch: Batch indices [num_nodes] (optional, for batched graphs)
            
        Returns:
            Logits [batch_size, num_classes]
        """
        # Handle batched input [batch, 33, features]
        if x.dim() == 3:
            batch_size, num_nodes, num_features = x.shape
            x = x.view(-1, num_features)  # [batch*33, features]
            batch = torch.arange(batch_size, device=x.device).repeat_interleave(num_nodes)
        
        edge_index = self.edge_index.to(x.device)
        
        # Expand edge_index for batched graphs
        if batch is not None and batch.max() > 0:
            batch_size = batch.max().item() + 1
            edge_indices = []
            for b in range(batch_size):
                offset = b * 33
                edge_indices.append(edge_index + offset)
            edge_index = torch.cat(edge_indices, dim=1)
        
        # Input projection
        x = self.input_proj(x)
        
        # Apply GraphSAGE convolutions with residual connections
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            identity = x
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # Residual connection (skip first layer since dimensions may differ)
            if i > 0:
                x = x + identity
        
        # Global pooling
        x = global_mean_pool(x, batch)
        
        # Classification
        x = self.classifier(x)
        return x


class CoachStyleEncoder(nn.Module):
    """Encode coach-specific style information.
    
    Learns embeddings for different coaches and techniques, allowing
    the model to capture style-specific nuances in technique execution.
    """
    
    def __init__(self, 
                 num_coaches: int = 10,
                 embedding_dim: int = 64,
                 num_techniques: int = 50,
                 technique_dim: int = 32,
                 output_dim: int = 64):
        """Initialize coach style encoder.
        
        Args:
            num_coaches: Number of different coaches
            embedding_dim: Coach embedding dimension
            num_techniques: Maximum number of techniques
            technique_dim: Technique embedding dimension
            output_dim: Output feature dimension
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for this model")
            
        super().__init__()
        
        # Coach embeddings
        self.coach_embedding = nn.Embedding(num_coaches, embedding_dim)
        
        # Technique embeddings
        self.technique_embedding = nn.Embedding(num_techniques, technique_dim)
        
        # Style fusion network
        self.style_fusion = nn.Sequential(
            nn.Linear(embedding_dim + technique_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU()
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize embedding weights."""
        nn.init.normal_(self.coach_embedding.weight, mean=0, std=0.02)
        nn.init.normal_(self.technique_embedding.weight, mean=0, std=0.02)
    
    def forward(self, coach_idx: torch.Tensor, technique_idx: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            coach_idx: Coach indices [batch_size]
            technique_idx: Technique indices [batch_size]
            
        Returns:
            Style features [batch_size, output_dim]
        """
        # Get embeddings
        coach_emb = self.coach_embedding(coach_idx)
        technique_emb = self.technique_embedding(technique_idx)
        
        # Combine embeddings
        combined = torch.cat([coach_emb, technique_emb], dim=-1)
        
        # Fuse style features
        style_features = self.style_fusion(combined)
        return style_features


class TemporalAttentionBlock(nn.Module):
    """Self-attention block for temporal sequences.
    
    Allows the model to attend to different parts of the movement
    sequence when making assessments.
    """
    
    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for this model")
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with residual connections.
        
        Args:
            x: Input tensor [batch, seq_len, hidden_dim]
            mask: Attention mask (optional)
            
        Returns:
            (output, attention_weights)
        """
        # Self-attention with residual
        attn_out, attn_weights = self.attention(x, x, x, key_padding_mask=mask)
        x = self.norm1(x + attn_out)
        
        # FFN with residual
        x = self.norm2(x + self.ffn(x))
        
        return x, attn_weights


class FormAssessor(nn.Module):
    """Assess form quality and provide corrective feedback.
    
    Uses LSTM for temporal processing combined with multi-head attention
    to identify key moments in technique execution and assess quality
    across multiple aspects.
    
    Features:
    - Bidirectional LSTM for temporal context
    - Multi-head attention for key moment identification
    - Separate assessment heads for different aspects
    - Optional biomechanics regression
    """
    
    ASPECT_NAMES = ['speed', 'force', 'timing', 'balance', 'coordination']
    
    def __init__(self, 
                 pose_dim: int = 4,  # x, y, z, visibility per landmark
                 hidden_dim: int = 256,
                 num_landmarks: int = 33,
                 num_aspects: int = 5,
                 num_attention_heads: int = 8,
                 num_lstm_layers: int = 2,
                 dropout: float = 0.3,
                 bidirectional: bool = True):
        """Initialize form assessor.
        
        Args:
            pose_dim: Dimension of each landmark
            hidden_dim: Hidden layer dimension
            num_landmarks: Number of landmarks per pose
            num_aspects: Number of assessment aspects
            num_attention_heads: Number of attention heads
            num_lstm_layers: Number of LSTM layers
            dropout: Dropout rate
            bidirectional: Use bidirectional LSTM
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for this model")
            
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_aspects = num_aspects
        self.bidirectional = bidirectional
        
        # Input projection
        input_dim = num_landmarks * pose_dim
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Bidirectional LSTM for temporal processing
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Adjust dimension for bidirectional
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # Project LSTM output back to hidden_dim for attention
        self.lstm_proj = nn.Linear(lstm_output_dim, hidden_dim)
        
        # Temporal attention block
        self.attention_block = TemporalAttentionBlock(
            hidden_dim=hidden_dim,
            num_heads=num_attention_heads,
            dropout=dropout
        )
        
        # Assessment heads (one per aspect)
        self.aspect_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid(),
                Scale(10.0)  # Scale to 0-10
            ) for _ in range(num_aspects)
        ])
        
        # Overall score head
        self.overall_head = nn.Sequential(
            nn.Linear(hidden_dim + num_aspects, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
            Scale(10.0)  # Scale to 0-10
        )
        
        # Optional: Biomechanics regression head
        self.biomechanics_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 6),  # 6 biomechanics metrics
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.kaiming_normal_(param)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
    
    def forward(self, poses: torch.Tensor, 
                return_biomechanics: bool = False) -> Dict[str, torch.Tensor]:
        """Forward pass.
        
        Args:
            poses: Pose sequence [batch_size, seq_len, num_landmarks, pose_dim]
            return_biomechanics: Also return biomechanics predictions
            
        Returns:
            Dictionary with assessment scores and features
        """
        batch_size, seq_len, num_landmarks, pose_dim = poses.shape
        
        # Flatten landmarks per frame
        x = poses.view(batch_size, seq_len, -1)  # [batch, seq_len, num_landmarks * pose_dim]
        
        # Input projection
        x = self.input_proj(x)  # [batch, seq_len, hidden_dim]
        
        # LSTM processing
        lstm_out, _ = self.lstm(x)  # [batch, seq_len, hidden_dim * 2] if bidirectional
        
        # Project back to hidden_dim
        lstm_out = self.lstm_proj(lstm_out)  # [batch, seq_len, hidden_dim]
        
        # Attention
        attn_out, attn_weights = self.attention_block(lstm_out)
        
        # Pool over time (weighted by attention)
        # Use mean pooling as fallback
        pooled = attn_out.mean(dim=1)  # [batch, hidden_dim]
        
        # Assess each aspect
        aspect_scores = []
        for head in self.aspect_heads:
            score = head(pooled)
            aspect_scores.append(score)
        
        aspect_tensor = torch.cat(aspect_scores, dim=1)  # [batch, num_aspects]
        
        # Overall assessment (considers aspect scores)
        overall_input = torch.cat([pooled, aspect_tensor], dim=1)
        overall_score = self.overall_head(overall_input)  # [batch, 1]
        
        result = {
            'aspect_scores': aspect_tensor,  # [batch, num_aspects]
            'overall_score': overall_score,  # [batch, 1]
            'features': pooled,  # [batch, hidden_dim]
            'attention_weights': attn_weights,  # [batch, num_heads, seq_len, seq_len]
        }
        
        if return_biomechanics:
            result['biomechanics'] = self.biomechanics_head(pooled)
        
        return result
    
    def get_aspect_names(self) -> List[str]:
        """Get names of assessment aspects."""
        return self.ASPECT_NAMES[:self.num_aspects]


class TechniqueClassifier(nn.Module):
    """Combined model for technique classification and assessment.
    
    Combines GraphSAGE for spatial features and LSTM for temporal features
    to both classify techniques and assess their quality.
    """
    
    def __init__(self,
                 num_techniques: int = 50,
                 num_coaches: int = 10,
                 hidden_dim: int = 256,
                 num_landmarks: int = 33,
                 pose_dim: int = 4):
        """Initialize technique classifier.
        
        Args:
            num_techniques: Number of technique classes
            num_coaches: Number of coaches
            hidden_dim: Hidden dimension
            num_landmarks: Number of pose landmarks
            pose_dim: Features per landmark
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for this model")
            
        super().__init__()
        
        # Spatial feature extractor (per-frame)
        self.spatial_encoder = GraphSAGEModel(
            num_features=pose_dim,
            hidden_channels=hidden_dim,
            num_classes=hidden_dim,  # Use as feature extractor
            num_layers=2
        )
        
        # Temporal feature extractor
        self.temporal_encoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )
        
        # Coach style encoder
        self.style_encoder = CoachStyleEncoder(
            num_coaches=num_coaches,
            num_techniques=num_techniques,
            output_dim=64
        )
        
        # Classification head
        combined_dim = hidden_dim * 2 + 64  # BiLSTM + style
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_techniques)
        )
        
        # Biomechanics regression head
        self.biomechanics_head = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 6)  # 6 biomechanics metrics
        )
    
    def forward(self, 
                poses: torch.Tensor,
                coach_idx: Optional[torch.Tensor] = None,
                technique_idx: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            poses: [batch, seq_len, num_landmarks, pose_dim]
            coach_idx: [batch] coach indices (optional)
            technique_idx: [batch] technique indices for style (optional)
            
        Returns:
            (classification_logits, biomechanics_predictions)
        """
        batch_size, seq_len, num_landmarks, pose_dim = poses.shape
        
        # Extract spatial features per frame
        frame_features = []
        for t in range(seq_len):
            frame = poses[:, t]  # [batch, num_landmarks, pose_dim]
            feat = self.spatial_encoder(frame)  # [batch, hidden_dim]
            frame_features.append(feat)
        
        spatial_features = torch.stack(frame_features, dim=1)  # [batch, seq_len, hidden_dim]
        
        # Extract temporal features
        temporal_out, (h_n, _) = self.temporal_encoder(spatial_features)
        
        # Use final hidden states from both directions
        temporal_features = torch.cat([h_n[-2], h_n[-1]], dim=1)  # [batch, hidden_dim * 2]
        
        # Get style features if available
        if coach_idx is not None and technique_idx is not None:
            style_features = self.style_encoder(coach_idx, technique_idx)
        else:
            style_features = torch.zeros(batch_size, 64, device=poses.device)
        
        # Combine features
        combined = torch.cat([temporal_features, style_features], dim=1)
        
        # Classification and biomechanics
        class_logits = self.classifier(combined)
        biomechanics = self.biomechanics_head(combined)
        
        return class_logits, biomechanics

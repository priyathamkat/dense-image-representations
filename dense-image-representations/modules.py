import torch
from torch import nn


class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        projection_dim: int,
        dropout: float
    ):
        """
        ProjectionHead is a module that applies projection and non-linear transformations to input tensors.

        Args:
            embedding_dim (int): The dimension of the input tensor.
            projection_dim (int): The dimension of the projected tensor.
            dropout (float): The dropout probability.

        """
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x: torch.Tensor):
        """
        Forward pass of the ProjectionHead module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying projection and non-linear transformations.

        """
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x

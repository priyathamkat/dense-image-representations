import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import T5EncoderModel
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class MLP(nn.Module):
    def __init__(self, dim, num_layers=3) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(dim, dim) for _ in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = F.relu(x)
        return x


class ProjectionHead(nn.Module):
    def __init__(self, embedding_dim: int, projection_dim: int, dropout: float):
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


class VisionLanguageEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        projection_dim: int,
        transformer_width: int,
        transformer_heads: int,
        transformer_layers: int,
        context_length: int = 77,
        embed_edges: bool = False,
        #  vocab_size: int = 49408,
    ):
        super().__init__()

        self.context_length = context_length
        self.transformer_heads = transformer_heads

        self.image_transformer = TransformerEncoder(
            encoder_layer=TransformerEncoderLayer(
                d_model=transformer_width, nhead=transformer_heads
            ),
            num_layers=transformer_layers,
        )

        self.text_transformer = T5EncoderModel.from_pretrained("google-t5/t5-small")

        # self.vocab_size = vocab_size
        # self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(
            torch.empty(self.context_length, transformer_width)
        )
        self.ln_final = nn.LayerNorm(transformer_width)

        # self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        # self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.image_projection = ProjectionHead(
            embedding_dim=embed_dim, projection_dim=projection_dim, dropout=0.1
        )
        self.text_projection = ProjectionHead(
            embedding_dim=embed_dim, projection_dim=projection_dim, dropout=0.1
        )

        self.edge_embedding = None
        if embed_edges:
            self.edge_embedding = MLP(transformer_width)

        self.initialize_parameters()

    def initialize_parameters(self):
        # nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        # proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        # attn_std = self.transformer.width ** -0.5
        # fc_std = (2 * self.transformer.width) ** -0.5
        # for block in self.transformer.resblocks:
        #     nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
        #     nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
        #     nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
        #     nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        # if self.text_projection is not None:
        #     nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def encode_image(self, image_tokens, image_attention_mask, num_nodes: int = None):
        # x = self.token_embedding(image_tokens)  # [batch_size, n_ctx, d_model]
        if self.edge_embedding and num_nodes is not None:
            image_tokens["edges"] = self.edge_embedding(image_tokens["edges"])

        x = torch.cat([image_tokens["edges"], image_tokens["nodes"]], dim=0)
        # x = x + self.positional_embedding  # TODO: Add smarter positional embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.image_transformer(
            x, mask=image_attention_mask.repeat(self.transformer_heads, 1, 1)
        )
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def encode_text(self, text_tokens):
        return self.text_transformer(text_tokens).last_hidden_state

    def forward(self, image_tokens, image_attention_mask, text_tokens):
        image_features = self.encode_image(image_tokens, image_attention_mask)
        text_features = self.encode_text(text_tokens)

        image_embeddings = self.image_projection(image_features)
        text_embeddings = self.text_projection(text_features)

        return image_embeddings, text_embeddings

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import T5EncoderModel, ViTModel
from torch.nn import TransformerEncoder, TransformerEncoderLayer

import pdb

class MLP(nn.Module):
    def __init__(self, in_features, out_features, num_layers=3) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Conv1d(in_features, in_features, 1) for _ in range(num_layers - 1)]
        )
        self.final = nn.Conv1d(in_features, out_features, 1)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = F.relu(x)
        x = self.final(x)
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


EDGES = "edges"
IMAGE_EMBEDDING = "image_embedding"
NODES = "nodes"


class VisionLanguageEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        projection_dim: int,
        transformer_width: int,
        transformer_heads: int,
        transformer_layers: int,
        image_embedding_size: int,
        context_length: int = 77,
        preembed_nodes: bool = False,
        #  vocab_size: int = 49408,
    ):
        super().__init__()

        self.context_length = context_length
        self.transformer_heads = transformer_heads

        self.image_transformer = TransformerEncoder(
            encoder_layer=TransformerEncoderLayer(
                d_model=transformer_width,
                nhead=transformer_heads,
                batch_first=True,
            ),
            num_layers=transformer_layers,
        )

        self.text_transformer = T5EncoderModel.from_pretrained("google-t5/t5-small")

        # self.vocab_size = vocab_size
        # self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embeddings = nn.ParameterDict(
            {
                EDGES: nn.Parameter(torch.empty((transformer_width,))),
                NODES: nn.Parameter(torch.empty((transformer_width,))),
            }
        )
        if not preembed_nodes:
            self.positional_embeddings[IMAGE_EMBEDDING] = nn.Parameter(
                torch.empty((transformer_width,))
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

        self.preembed_nodes = preembed_nodes
        if self.preembed_nodes:
            self.image_embedder = MLP(
                transformer_width + image_embedding_size, transformer_width
            )
        else:
            self.image_embedder = MLP(image_embedding_size, transformer_width)

        self.initialize_parameters()

    def initialize_parameters(self):
        # nn.init.normal_(self.token_embedding.weight, std=0.02)
        for _, v in self.positional_embeddings.items():
            nn.init.normal_(v, std=0.01)

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

    def encode_image(
        self,
        tokens: torch.Tensor,
        image_embedding: torch.Tensor,
        num_non_pad_tokens: list[int],
        num_nodes: list[int],
        image_attention_mask: torch.Tensor = None,
    ):
        """Expects image_tokens to be a dictionary with three keys: 'edges', 'nodes' and 'image_embedding'.
        The shapes of the values must be the following:

        'tokens': batch_size x max_seq_len x transformer_width
        num_nodes + num_edges == num_non_pad_tokens

        'image_embedding': batch_size x image_embedding_size
        """

        image_embedding = image_embedding.unsqueeze(1)
        max_num_nodes = max(num_nodes)
        nodes = tokens[:, :max_num_nodes, :]

        if self.preembed_nodes:
            nodes = self.image_embedder(
                torch.cat(
                    [nodes, image_embedding.repeat(1, max_num_nodes, 1)], dim=-1
                ).permute(0, 2, 1)
            ).permute(0, 2, 1)
            for i, n in enumerate(num_nodes):
                tokens[i, :n] = nodes[i, :n]
        else:
            image_embedding = self.image_embedder(
                image_embedding.permute(0, 2, 1)
            ).permute(0, 2, 1)

        for i, (n, p) in enumerate(zip(num_nodes, num_non_pad_tokens)):
            tokens[i, :n] = tokens[i, :n] + self.positional_embeddings[NODES]
            tokens[i, n:p] = tokens[i, n:p] + self.positional_embeddings[EDGES]

        if self.preembed_nodes:
            x = tokens
        else:
            image_embedding = (
                image_embedding + self.positional_embeddings[IMAGE_EMBEDDING]
            )
            x = torch.cat(
                [image_embedding, tokens],
                dim=1,
            )
            x = x[:, : self.context_length, :]
        x = self.image_transformer(
            x, mask=image_attention_mask
        )
        x = self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def encode_text(self, text_tokens):
        return self.text_transformer(text_tokens).last_hidden_state

    def forward(self, 
                image_tokens,
                image_features,
                num_non_pad_tokens,
                num_nodes,
                text_tokens,
                image_attention_mask = None):
        image_features = self.encode_image(image_tokens,
                                            image_features,
                                            num_non_pad_tokens,
                                            num_nodes,
                                            image_attention_mask)
        text_features = self.encode_text(text_tokens)

        image_embeddings = self.image_projection(image_features)
        text_embeddings = self.text_projection(text_features)

        return image_embeddings, text_embeddings



class VisionLanguageEncoderBase(nn.Module):
    def __init__(
        self,
        image_embed_dim: int,
        text_embed_dim: int,
        projection_dim: int,
    ):
        super().__init__()

        self.image_encoder = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k').to('cuda')

        self.text_transformer = T5EncoderModel.from_pretrained("google-t5/t5-small")

        self.image_projection = ProjectionHead(
            embedding_dim=image_embed_dim, projection_dim=projection_dim, dropout=0.1
        )
        self.text_projection = ProjectionHead(
            embedding_dim=text_embed_dim, projection_dim=projection_dim, dropout=0.1
        )

    def encode_image(
        self,
        images,
    ):
        x = self.image_encoder(images).pooler_output
        return x

    def encode_text(self, text_tokens):
        return self.text_transformer(text_tokens).last_hidden_state

    def forward(self, 
                images,
                text_tokens,):
        image_features = self.encode_image(images)
        text_features = self.encode_text(text_tokens)

        image_embeddings = self.image_projection(image_features)
        text_embeddings = self.text_projection(text_features)

        return image_embeddings, text_embeddings

    
if __name__ == "__main__":
    vle = VisionLanguageEncoder(16, 16, 16, 8, 6, 32, preembed_nodes=True)
    num_nodes = [1, 2, 3, 4, 5]
    num_non_pad_tokens = [10, 11, 12, 13, 14]
    tokens = torch.randn(5, 77, 16)
    image_embedding = torch.randn(5, 32)
    print(
        vle.encode_image(
            tokens,
            image_embedding,
            num_non_pad_tokens,
            num_nodes,
            torch.randn(5, 77, 77),
        ).shape
    )

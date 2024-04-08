import torch
from torch import nn
import torch.nn.functional as F


class ContrastiveLoss:
    """
    Contrastive loss function for training dense image representations.

    Args:
        temperature (float): The temperature parameter for controling the sharpness of the probability distribution produced by the softmax function.
    """

    def __init__(
        self,
        temperature: float = 1.0,
    ):
        self.temperature = temperature
        self._cross_entropy = nn.CrossEntropyLoss(reduction='none')
    
    def __call__(
        self,
        text_embeddings: torch.Tensor,
        image_embeddings: torch.Tensor,
    ):
        """
        Compute the contrastive loss between text and image embeddings.

        Args:
            text_embeddings (torch.Tensor): Tensor of shape (batch_size, embedding_dim) representing text embeddings.
            image_embeddings (torch.Tensor): Tensor of shape (batch_size, embedding_dim) representing image embeddings.

        Returns:
            torch.Tensor: The computed contrastive loss.
        """
        assert text_embeddings.shape == image_embeddings.shape

        logits = (text_embeddings @ image_embeddings.T) / self.temperature

        images_similarity = image_embeddings @ image_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T

        targets = F.softmax(
            (images_similarity + texts_similarity) / 2 * self.temperature, dim=-1
        )

        texts_loss = self._cross_entropy(logits, torch.argmax(targets, dim=1))
        images_loss = self._cross_entropy(logits.T, torch.argmax(targets.T, dim=1))

        loss =  (images_loss + texts_loss) / 2.0

        return loss.mean()

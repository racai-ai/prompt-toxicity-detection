"""Shared loss functions for NER models."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MulticlassDiceLoss(nn.Module):
    """Dice loss for multiclass token classification (NER).

    Weights the overlap between prediction and target.
    """

    def __init__(self, smooth: float = 1e-6, ignore_index: int = -100) -> None:
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute multiclass Dice loss.

        Parameters
        ----------
        logits:
            Shape ``[batch_size, seq_len, num_classes]``.
        targets:
            Shape ``[batch_size, seq_len]``.
        """
        probs = F.softmax(logits, dim=-1)
        num_classes = logits.shape[-1]

        mask = (targets != self.ignore_index)
        targets_masked = targets[mask]
        probs_masked = probs[mask]

        targets_one_hot = F.one_hot(targets_masked, num_classes).float()

        intersection = torch.sum(probs_masked * targets_one_hot, dim=0)
        cardinality = torch.sum(probs_masked + targets_one_hot, dim=0)

        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice_score.mean()

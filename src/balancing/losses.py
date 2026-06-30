"""
Custom loss functions for handling class imbalance.

Implements:
  - Weighted Cross-Entropy Loss (class weights inversely proportional to frequency)
  - Focal Loss (Lin et al., 2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_class_weights(labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Compute inverse-frequency class weights from labeled data.

    Args:
        labels: Node labels tensor.
        mask: Boolean mask for which nodes to consider.

    Returns:
        Tensor of shape [num_classes] with weights.
    """
    masked_labels = labels[mask]
    classes = torch.unique(masked_labels[masked_labels >= 0])
    counts = torch.zeros(len(classes))
    for i, c in enumerate(classes):
        counts[i] = (masked_labels == c).sum().float()
    weights = counts.sum() / (len(classes) * counts)
    return weights


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance (Lin et al., 2017).

    Reduces the contribution of easy-to-classify samples and focuses
    training on hard examples near the decision boundary.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha: Weighting factor for the RARE (positive, class=1) class.
               Must be in (0, 1). Values > 0.5 up-weight the rare class
               (standard usage for imbalanced data). Can also be a tensor
               of per-class weights [w_class0, w_class1] for full control.
               Default 0.75 = rare class gets 3× the weight of majority.
        gamma: Focusing parameter. gamma=0 reduces to weighted CE.
               Higher gamma = more focus on hard examples.
        reduction: 'mean', 'sum', or 'none'.

    Note: Semantics change in v3 (2026-04). Previously alpha meant "class 1
    weight" directly; that behavior mapped alpha=0.25 to DOWN-weighting
    the rare class (reverse of what imbalanced data needs). Now alpha is
    the weight for the rare class, matching intuition and literature.
    """

    def __init__(
        self,
        alpha: float | torch.Tensor = 0.75,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

        if isinstance(alpha, (float, int)):
            # alpha = weight for rare class (class 1), (1-alpha) for majority (class 0)
            self.alpha = torch.tensor([1 - alpha, alpha])
        else:
            self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Raw model output [N, C].
            targets: Ground truth labels [N].

        Returns:
            Focal loss value.
        """
        probs = F.softmax(logits, dim=-1)
        targets_one_hot = F.one_hot(targets, num_classes=logits.size(-1)).float()

        # Gather probabilities for true class
        p_t = (probs * targets_one_hot).sum(dim=-1)

        # Compute focal weight
        focal_weight = (1 - p_t) ** self.gamma

        # Compute cross-entropy
        ce_loss = F.cross_entropy(logits, targets, reduction="none")

        # Apply alpha weighting per class
        alpha = self.alpha.to(logits.device)
        alpha_t = alpha[targets]

        # Final focal loss
        loss = alpha_t * focal_weight * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


def get_loss_function(
    technique: str,
    labels: torch.Tensor = None,
    mask: torch.Tensor = None,
    gamma: float = 2.0,
    alpha: float = 0.75,
    device: str = "cpu",
) -> nn.Module:
    """
    Factory function to get the appropriate loss function.

    Args:
        technique: One of "none", "class_weighting", "focal_loss".
        labels: Node labels (needed for weight computation).
        mask: Training mask (needed for weight computation).
        gamma: Focal loss gamma parameter.
        alpha: Focal loss alpha parameter.
        device: Target device.

    Returns:
        Loss function module.
    """
    if technique == "none":
        return nn.CrossEntropyLoss()
    elif technique == "class_weighting":
        weights = compute_class_weights(labels, mask).to(device)
        print(f"  Class weights: {weights.tolist()}")
        return nn.CrossEntropyLoss(weight=weights)
    elif technique == "focal_loss":
        print(f"  Focal Loss: gamma={gamma}, alpha={alpha}")
        return FocalLoss(alpha=alpha, gamma=gamma)
    elif technique == "graphsmote":
        # GraphSMOTE operates at graph level (node augmentation), not at loss level.
        # Return standard CE; augmentation must be applied before training.
        print(f"  GraphSMOTE: using standard CrossEntropyLoss (graph augmentation is separate)")
        return nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unknown balancing technique: {technique}")

import torch
from torch import nn
import torch.nn.functional as F


class BinaryDiceLoss(nn.Module):
    """Dice loss for binary classification.

    Args:
        smooth: A float to avoid NaN errors, default: 1.
        p: The power used in the denominator, default: 2.
        reduction: How to reduce the loss ('mean', 'sum', or 'none').
    """

    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        # Ensure the batch sizes of predict and target match
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"

        # Flatten the input tensors
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        # Compute the Dice loss
        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth  # Intersection
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth  # Union

        loss = 1 - num / den  # Dice loss calculation

        # Apply reduction method
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class DiceLoss(nn.Module):
    """Dice loss for multi-class segmentation.

    Args:
        n_classes: Number of classes for segmentation.
        weights: Optional list of weights for each class.
        softmax: Whether to apply softmax to the inputs.
    """

    def __init__(self, n_classes, weights=None, softmax=True):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes
        self.weights = weights if weights is not None else [1] * self.n_classes  # Default weights
        self.softmax = softmax

    def _one_hot_encoder(self, input_tensor):
        """Convert class indices to one-hot encoded format."""
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # Create a binary mask for class i
            tensor_list.append(temp_prob.unsqueeze(1))  # Add a new dimension
        output_tensor = torch.cat(tensor_list, dim=1)  # Concatenate along the class dimension
        return output_tensor.float()  # Return as float tensor

    def _dice_loss(self, score, target):
        """Calculate Dice loss for a single class."""
        target = target.float()
        smooth = 1e-5  # Smoothing factor
        intersect = torch.sum(score * target)  # Intersection
        y_sum = torch.sum(target * target)  # Ground truth area
        z_sum = torch.sum(score * score)  # Predicted area
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)  # Dice coefficient
        return 1 - loss  # Return the loss

    def forward(self, inputs, target):
        """Calculate the total Dice loss across all classes."""
        if self.softmax:
            inputs = torch.softmax(inputs, dim=1)  # Apply softmax if needed
        target = self._one_hot_encoder(target)  # Convert target to one-hot encoding

        # Ensure the shapes of inputs and targets match
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(),
                                                                                                  target.size())

        loss = 0.0
        for i in range(self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])  # Calculate Dice loss for each class
            loss += dice * self.weights[i]  # Weighted loss accumulation

        return loss / self.n_classes  # Average loss across classes


class IoULoss(nn.Module):
    """IoU loss for multi-class segmentation.

    Args:
        n_classes: Number of classes for segmentation.
        weights: Optional list of weights for each class.
        softmax: Whether to apply softmax to the inputs.
    """

    def __init__(self, n_classes, weights=None, softmax=True):
        super(IoULoss, self).__init__()
        self.n_classes = n_classes
        self.weights = weights if weights is not None else [1] * self.n_classes  # Default weights
        self.softmax = softmax

    def _one_hot_encoder(self, input_tensor):
        """Convert class indices to one-hot encoded format."""
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # Create a binary mask for class i
            tensor_list.append(temp_prob.unsqueeze(1))  # Add a new dimension
        output_tensor = torch.cat(tensor_list, dim=1)  # Concatenate along the class dimension
        return output_tensor.float()  # Return as float tensor

    def _iou_loss(self, score, target):
        """Calculate IoU loss for a single class."""
        target = target.float()
        smooth = 1e-5  # Smoothing factor
        intersect = torch.sum(score * target)  # Intersection
        y_sum = torch.sum(target * target)  # Ground truth area
        z_sum = torch.sum(score * score)  # Predicted area
        loss = (intersect + smooth) / (z_sum + y_sum - intersect + smooth)  # IoU
        return 1 - loss  # Return the loss

    def forward(self, inputs, target):
        """Calculate the total IoU loss across all classes."""
        if self.softmax:
            inputs = torch.softmax(inputs, dim=1)  # Apply softmax if needed
        target = self._one_hot_encoder(target)  # Convert target to one-hot encoding

        # Ensure the shapes of inputs and targets match
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(),
                                                                                                  target.size())

        loss = 0.0
        for i in range(self.n_classes):
            iou = self._iou_loss(inputs[:, i], target[:, i])  # Calculate IoU loss for each class
            loss += iou * self.weights[i]  # Weighted loss accumulation

        return loss / self.n_classes  # Average loss across classes


class ERS(nn.Module):
    """Combined loss: Cross-Entropy + IoU loss + Entropy regularization.

    Args:
        alpha: Weight for IoU loss in the final loss calculation.
        smooth: Smoothing factor for numerical stability (not actively used).
    """

    def __init__(self, alpha=0.5, smooth=1e-6):
        super(ERS, self).__init__()
        self.IoULoss = IoULoss(n_classes=2)  # Initialize IoU loss for binary classification
        self.cross_entropy_loss = nn.CrossEntropyLoss()  # Initialize Cross-Entropy loss
        self.alpha = alpha  # Weight for IoU loss

    def forward(self, outputs, targets):
        """Calculate the combined loss."""
        # Assuming outputs are logits (before softmax)
        # and targets are class indices for Cross-Entropy
        ce_loss = self.cross_entropy_loss(outputs, targets)  # Calculate cross-entropy loss

        prob = F.softmax(outputs, dim=1)  # Get probabilities from logits
        # Calculate entropy
        entropy = -torch.sum(prob * torch.log(prob + 1e-9), dim=1).mean()  # Avoid log(0) with a small epsilon

        # Calculate IoU loss
        iou_loss = self.IoULoss(outputs, targets)

        # Combine losses
        return self.alpha * iou_loss + (1 - self.alpha) * ce_loss + 0.2 * entropy  # Return total loss

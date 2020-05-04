# Code taken from the following repository on GitHub:
# https://github.com/bonlime/pytorch-tools


from enum import Enum

import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss


def soft_dice_score(y_pred, y_true, dims=None, eps=1e-4):
    """
    `Soft` means than when `y_pred` and `y_true` are zero this function will
    return 1, while in many other implementations it will return 0.
    Args:
        y_pred (torch.Tensor): Of shape `NxCx*` where * means any
            number of additional dimensions
        y_true (torch.Tensor): `NxCx*`, same shape as `y_pred`
        dims (Tuple[int], optional): Dims to use for calculating
        eps (float): Laplace smoothing
    """
    if y_pred.size() != y_true.size():
        raise ValueError("Input and target shapes should match")

    if dims is not None:
        intersection = torch.sum(y_pred * y_true, dim=dims)
        cardinality = torch.sum(y_pred + y_true, dim=dims)
    else:
        intersection = torch.sum(y_pred * y_true)
        cardinality = torch.sum(y_pred + y_true)
    dice_score = (2.0 * intersection + eps) / (cardinality + eps)
    return dice_score


class Mode(Enum):
    BINARY = "binary"
    MULTICLASS = "multiclass"
    MULTILABEL = "multilabel"


class Loss(_Loss):
    """Loss which supports addition and multiplication"""

    def __add__(self, other):
        if isinstance(other, Loss):
            return SumOfLosses(self, other)
        else:
            raise ValueError("Loss should be inherited from `Loss` class")

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, value):
        if isinstance(value, (int, float)):
            return WeightedLoss(self, value)
        else:
            raise ValueError("Loss should be multiplied by int or float")

    def __rmul__(self, other):
        return self.__mul__(other)


class WeightedLoss(Loss):
    """
    Wrapper class around loss function that applies weighted with fixed factor.
    This class helps to balance multiple losses if they have different scales
    """

    def __init__(self, loss, weight=1.0):
        super().__init__()
        self.loss = loss
        self.weight = torch.Tensor([weight])

    def forward(self, *inputs):
        l = self.loss(*inputs)
        self.weight = self.weight.to(l.device)
        return l * self.weight[0]


class SumOfLosses(Loss):
    def __init__(self, l1, l2):
        super().__init__()
        self.l1 = l1
        self.l2 = l2

    def __call__(self, *inputs):
        return self.l1(*inputs) + self.l2(*inputs)


class DiceLoss(Loss):
    """
    Implementation of Dice loss for image segmentation task.
    It supports binary, multiclass and multilabel cases
    Args:
        mode (str): Target mode {'binary', 'multiclass', 'multilabel'}
            'multilabel' - expects y_true of shape [N, C, H, W]
            'multiclass', 'binary' - expects y_true of shape [N, H, W]
        log_loss (bool): If True, loss computed as `-log(jaccard)`; otherwise `1 - jaccard`
        from_logits (bool): If True assumes input is raw logits
        eps (float): small epsilon for numerical stability
    Shape:
        y_pred: [N, C, H, W]
        y_true: [N, C, H, W] or [N, H, W] depending on mode
    """

    IOU_FUNCTION = soft_dice_score

    def __init__(self, mode="binary", log_loss=False, from_logits=False, eps=1.0):
        super(DiceLoss, self).__init__()
        self.mode = Mode(mode)  # raises an error if not valid
        self.log_loss = log_loss
        self.from_logits = from_logits
        self.eps = eps

    def forward(self, y_pred, y_true):
        if self.from_logits:
            # Apply activations to get [0..1] class probabilities
            if self.mode == Mode.BINARY or self.mode == Mode.MULTILABEL:
                y_pred = y_pred.sigmoid()
            elif self.mode == Mode.MULTICLASS:
                y_pred = y_pred.softmax(dim=1)

        bs = y_true.size(0)
        num_classes = y_pred.size(1)
        dims = (0, 2)

        if self.mode == Mode.BINARY:
            y_true = y_true.view(bs, 1, -1)
            y_pred = y_pred.view(bs, 1, -1)
        elif self.mode == Mode.MULTICLASS:
            y_true = y_true.view(bs, -1)
            y_pred = y_pred.view(bs, num_classes, -1)
            y_true = torch.nn.functional.one_hot(
                y_true, num_classes
            )  # N,H*W -> N,H*W, C
            y_true = y_true.permute(0, 2, 1)  # H, C, H*W
        elif self.mode == Mode.MULTILABEL:
            y_true = y_true.view(bs, num_classes, -1)
            y_pred = y_pred.view(bs, num_classes, -1)

        scores = self.__class__.IOU_FUNCTION(
            y_pred, y_true.type(y_pred.dtype), dims=dims, eps=self.eps
        )

        if self.log_loss:
            loss = -torch.log(scores)
        else:
            loss = 1 - scores

        # IoU loss is defined for non-empty classes
        # So we zero contribution of channel that does not have true pixels
        mask = y_true.sum(dims) > 0
        loss *= mask.float()

        return loss.mean()

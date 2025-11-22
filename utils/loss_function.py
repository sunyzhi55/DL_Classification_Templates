from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class FocalBCELoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        """
        Focal Loss for binary classification, based on BCELoss.

        :param alpha: A balancing factor for positive and negative examples.
        :param gamma: A focusing parameter to reduce the loss for well-classified examples.
        :param reduction: Specifies the method to reduce the loss across all examples.
                          Options: 'none', 'mean', 'sum'.
        """
        super(FocalBCELoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Sigmoid activation to get probabilities
        inputs = torch.sigmoid(inputs)
        print('inputs', inputs)

        # Compute the BCELoss
        bce_loss = - targets * torch.log(inputs) + (1 - targets) * torch.log(1 - inputs)
        print('bce_loss', torch.mean(bce_loss))

        # Compute the focal loss factor (1 - p_t)^gamma
        focal_factor = torch.pow(1 - (torch.exp(-bce_loss)), self.gamma)
        print('focal_factor', focal_factor)

        # Apply the focal factor and alpha weighting
        focal_loss = self.alpha * focal_factor * bce_loss
        print('focal_loss', focal_loss)

        # Reduce the loss (mean, sum, or no reduction)
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class FocalLoss(nn.Module):
    def __init__(self, device, alpha=0.25, gamma=2, reduction='mean', position_weight: Optional[Tensor] = None):
        """
        Focal Loss implementation.
        Args:
            alpha (float or list): Class weight. If list, it should have same length as the number of classes.
            gamma (float): Focusing parameter to adjust the rate at which easy examples are down-weighted.
            reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.position_weight = position_weight
        self.bceLoss = nn.BCEWithLogitsLoss(reduction=self.reduction, pos_weight=self.position_weight).to(device)

    def forward(self, inputs, targets):
        # Calculate cross entropy loss
        # ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        # ce_loss = - F.binary_cross_entropy_with_logits(inputs, targets, pos_weight=weight, reduction=self.reduction)

        # ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction=self.reduction,
        #                                              pos_weight=self.position_weight).to(self.device)


        ce_loss = self.bceLoss(inputs, targets)
        # print('ce_loss', ce_loss)

        # Get probabilities for the correct class
        pt = torch.exp(-ce_loss)
        # print('pt', pt)

        # Focal loss calculation
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        # print('focal_loss', focal_loss)

        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss


class FocalLossWithTwoClass(nn.Module):
    def __init__(self, alpha=0.3, gamma=2, reduction='mean'):
        """
        Focal Loss implementation.
        Args:
            alpha (float or list): Class weight. If list, it should have same length as the number of classes.
            gamma (float): Focusing parameter to adjust the rate at which easy examples are down-weighted.
            reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
        """
        super(FocalLossWithTwoClass, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Calculate cross entropy loss
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # Get probabilities for the correct class
        pt = torch.exp(-ce_loss)

        # Focal loss calculation
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss


# 示例
if __name__ == '__main__':
    # 假设 batch size 为 4
    inputs = torch.tensor([[0.9, 0.1], [ 0.6, 0.8]],  )  # 模型输出（logits）
    targets = torch.tensor([[1.0, 0.0], [ 1.0, 1.0]])  # 真实标签
    print("inputs", inputs.shape)
    print("targets", targets.shape)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # criterion = FocalBCELoss(alpha=0.25, gamma=2, reduction='mean')

    pos_weight = torch.tensor([1.0, 5.1])  # 正类的权重可以根据类别不平衡情况调整

    # criterion = FocalLoss(device=device, alpha=0.25, gamma=2, reduction='mean', position_weight=pos_weight)
    criterion = FocalBCELoss(alpha=0.25, gamma=2, reduction='mean')
    loss = criterion(inputs, targets)
    print(f"Focal Loss: {loss.item()}")

    # 模拟模型输出（logits）
    # inputs = torch.tensor([0.9, 0.1, 0.6, 0.8])  # 未经过 Sigmoid 的原始输出
    # targets = torch.tensor([1.0, 0.0, 1.0, 1.0])  # 真实标签

    # 设置 pos_weight 来加权正类损失
    # pos_weight = torch.tensor([1.0])  # 正类的权重可以根据类别不平衡情况调整

    # 使用 BCEWithLogitsLoss 计算损失
    # criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(inputs, targets)
    print(f"BCEWithLogitsLoss with pos_weight: {loss.item()}")



# Focal Loss: 0.12624238431453705
# BCEWithLogitsLoss with pos_weight: 0.47353482246398926
import torch
import torch.nn as nn
import torch.nn.functional as F

class CategoricalFocalLoss(nn.Module):
    
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super(CategoricalFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

        if isinstance(alpha, (float, int, list)):
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        if isinstance(alpha, list):
            self.alpha = torch.tensor(alpha, dtype=torch.float32)

    def forward(self, inputs, targets):
        inputs = F.softmax(inputs, dim=1)
        
        targets_one_hot = F.one_hot(targets, num_classes=inputs.size(1)).float()
        
        log_p = inputs.log()
        ce_loss = -targets_one_hot * log_p
        p_t = inputs * targets_one_hot
        focal_loss = (1 - p_t) ** self.gamma * ce_loss

        if self.alpha is not None:
            if self.alpha.type() != inputs.data.type():
                self.alpha = self.alpha.type_as(inputs.data)
            alpha_t = self.alpha.gather(0, targets)
            focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
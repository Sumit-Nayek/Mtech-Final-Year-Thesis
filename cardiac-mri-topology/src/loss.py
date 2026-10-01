import torch
import torch.nn as nn
import torch.nn.functional as F
import config

class DirectionalGradientLoss(nn.Module):
    """
    Topology-Preserving Boundary Loss using non-trainable Sobel X/Y filters
    to penalize spatial edge inconsistencies and disjointed components.
    """
    def __init__(self, num_classes=config.NUM_SEG_CLASSES):
        super(DirectionalGradientLoss, self).__init__()
        self.num_classes = num_classes

        sobel_x = torch.tensor([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)

        sobel_y = torch.tensor([[-1, -2, -1],
                                [ 0,  0,  0],
                                [ 1,  2,  1]], dtype=torch.float32).view(1, 1, 3, 3)

        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def forward(self, preds, targets):
        preds_prob = F.softmax(preds, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        loss = 0.0
        for c in range(self.num_classes):
            pred_c = preds_prob[:, c:c+1, :, :]
            target_c = targets_one_hot[:, c:c+1, :, :]

            pred_grad_x = F.conv2d(pred_c, self.sobel_x, padding=1)
            pred_grad_y = F.conv2d(pred_c, self.sobel_y, padding=1)

            target_grad_x = F.conv2d(target_c, self.sobel_x, padding=1)
            target_grad_y = F.conv2d(target_c, self.sobel_y, padding=1)

            loss += F.mse_loss(pred_grad_x, target_grad_x) + F.mse_loss(pred_grad_y, target_grad_y)

        return loss / float(self.num_classes)

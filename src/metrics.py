# metrics.py
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice Loss class.
    """
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, predictions, targets, smooth=1):
        """
        Calculate the forward pass of the Dice loss function.
        Args:
            predictions (torch.Tensor): The predicted segmentation map.
            targets (torch.Tensor): The ground truth segmentation map.
            smooth (int, optional): A smoothing factor to prevent division by zero. Default is 1.
        Returns:
            torch.Tensor: The Dice loss of the predictions with respect to the targets.
        """
        # Apply sigmoid activation function 
        predictions = F.sigmoid(predictions)       
       
        # Flatten the prediction and target tensors
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        # Calculate intersection and dice score
        intersection = (predictions * targets).sum()                            
        dice_score = (2.*intersection + smooth)/(predictions.sum() + targets.sum() + smooth)  
        
        # Return dice loss
        return 1 - dice_score


class IoULoss(nn.Module):
    """
    Intersection over Union (IoU) Loss class.
    """
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        """
        Calculate the forward pass of the IoU loss function.
        Args:
            inputs (torch.Tensor): The predicted segmentation map.
            targets (torch.Tensor): The ground truth segmentation map.
            smooth (int, optional): A smoothing factor to prevent division by zero. Default is 1.
        Returns:
            torch.Tensor: The IoU loss of the predictions with respect to the targets.
        """
        
        # Apply sigmoid activation function 
        inputs = F.sigmoid(inputs)       
        
        # Flatten the prediction and target tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # Calculate intersection and union
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        IoU = (intersection + smooth)/(union + smooth)

        # Return IoU loss 
        return 1 - IoU

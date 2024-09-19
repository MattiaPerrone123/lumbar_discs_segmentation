import torch

def combined_dice_bce_loss(pred, target, bce_weight=0.5, dice_weight=0.5, epsilon=1e-6):
    #Calculate the combined Dice and Binary Cross-Entropy (BCE) loss.

    #BCE Loss
    bce_loss=torch.nn.functional.binary_cross_entropy_with_logits(pred, target)

    #Apply sigmoid to get probabilities for Dice loss calculation
    pred_probs=torch.sigmoid(pred)

    #Flatten the tensors to align vectors
    pred_flat=pred_probs.view(-1)
    target_flat=target.view(-1)

    #Dice Loss
    intersection=(pred_flat * target_flat).sum()
    dice_loss=1-(2. * intersection + epsilon) / (pred_flat.sum() + target_flat.sum() + epsilon)

    #Combined loss
    combined_loss=bce_weight * bce_loss + dice_weight * dice_loss

    return combined_loss
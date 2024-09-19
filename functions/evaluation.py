import torch
import scipy.stats as stats
import numpy as np


def get_iou_train(logits, targets):
    #Computing iou betweenn predicted masks and ground truth masks
    logits=torch.round(logits)
    targets=torch.round(targets)

    intersection=torch.logical_and(targets, logits).sum()
    union=torch.logical_or(targets, logits).sum()
    iou=intersection.item()/union.item()
    return iou


def get_dice(logits, targets):
    #Computing dice betweenn predicted masks and ground truth masks
    true_positives=torch.logical_and(targets, logits).sum().item()
    false_positives=(logits - targets).clamp(min=0).sum().item()
    false_negatives=(targets - logits).clamp(min=0).sum().item()

    dice=(2.0 * true_positives) / (2.0 * true_positives + false_positives + false_negatives)
    return dice


def calculate_ci(data, confidence_level=0.95):
    #Calculating confidence intervals
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    n = len(data)
    sem = std / np.sqrt(n)
    degrees_of_freedom = n - 1
    ci_low, ci_high = stats.t.interval(confidence_level, df=degrees_of_freedom, loc=mean, scale=sem)

    return ci_low, ci_high
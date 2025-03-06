import torch
import numpy as np
from sklearn.metrics import confusion_matrix

def calculate_metrics(pred, target):
    """Calculate various segmentation metrics"""
    pred = pred.view(-1)
    target = target.view(-1)
    
    # Convert to numpy for sklearn metrics
    pred_np = pred.cpu().numpy()
    target_np = target.cpu().numpy()
    
    # Calculate confusion matrix
    cm = confusion_matrix(target_np, pred_np)
    
    # Calculate metrics
    metrics = {}
    
    # Pixel Accuracy
    metrics['pixel_accuracy'] = (pred == target).float().mean().item()
    
    # IoU for each class
    intersection = torch.logical_and(pred, target)
    union = torch.logical_or(pred, target)
    iou = (intersection.sum().float() / union.sum().float()).item()
    metrics['iou'] = iou
    
    # Dice coefficient
    dice = (2 * intersection.sum().float() / (pred.sum() + target.sum()).float()).item()
    metrics['dice'] = dice
    
    return metrics

def save_metrics(metrics, filepath):
    """Save metrics to a file"""
    import json
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=4) 
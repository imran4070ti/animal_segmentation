import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns
from pathlib import Path
import pandas as pd

def plot_losses(train_losses, val_losses, save_dir):
    """Plot training and validation losses"""
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(Path(save_dir) / 'loss_curves.png')
    plt.close()

def visualize_prediction(image, prediction, save_path=None, mask=None):
    """
    Visualize prediction and optionally ground truth mask
    
    Args:
        image: Original image (numpy array)
        prediction: Model prediction mask
        save_path: Path to save visualization
        mask: Optional ground truth mask
    """
    if mask is not None:
        # If we have ground truth mask, show three images
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.imshow(image)
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(mask, cmap='gray')
        plt.title('Ground Truth')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(prediction, cmap='gray')
        plt.title('Prediction')
        plt.axis('off')
    else:
        # If we only have prediction, show two images
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(prediction, cmap='gray')
        plt.title('Prediction')
        plt.axis('off')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_confusion_matrix(cm, save_dir):
    """Plot confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(Path(save_dir) / 'confusion_matrix.png')
    plt.close()

# Alias for backward compatibility
plot_training_history = plot_losses 
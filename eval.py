import torch
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

from configs.config import Config
from models.unet import UNet
from preprocessing.dataset import AnimalSegmentationDataset
from preprocessing.transforms import Transforms
from utils.metrics import calculate_metrics

def evaluate(checkpoint_path):
    # Setup
    device = torch.device(Config.DEVICE)
    
    # Setup transforms
    transforms = Transforms(Config.IMAGE_SIZE)
    
    # Data
    val_dataset = AnimalSegmentationDataset(
        Config.DATA_ROOT, 
        split='val',
        transform=transforms.val_transform,
        target_size=Config.IMAGE_SIZE
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )
    
    # Model
    model = UNet(n_channels=3, n_classes=Config.NUM_CLASSES).to(device)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Evaluation
    metrics = {}
    total_metrics = {}
    
    with torch.no_grad():
        for images, masks in tqdm(val_loader):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)
            
            # Calculate batch metrics
            batch_metrics = calculate_metrics(predictions, masks)
            
            # Accumulate metrics
            for k, v in batch_metrics.items():
                total_metrics[k] = total_metrics.get(k, 0) + v
    
    # Average metrics
    num_batches = len(val_loader)
    metrics = {k: v / num_batches for k, v in total_metrics.items()}
    
    # Print metrics
    print("\nValidation Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    
    return metrics

if __name__ == '__main__':
    checkpoint_path = 'checkpoints/animal_seg_experiment/best_model.pth'
    metrics = evaluate(checkpoint_path) 
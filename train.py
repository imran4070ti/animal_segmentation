from configs.config import Config
from models.unet import UNet
from preprocessing.dataset import AnimalSegmentationDataset
from preprocessing.transforms import Transforms
from utils.trainer import Trainer
import torch
from torch.utils.data import DataLoader

def main():
    # Setup
    device = torch.device(Config.DEVICE)
    transforms = Transforms(
        target_size=Config.IMAGE_SIZE,
        use_augmentation=Config.USE_AUGMENTATION
    )
    
    # Data
    train_dataset = AnimalSegmentationDataset(
        Config.DATA_ROOT, 
        split='train',
        transform=transforms.train_transform,
        target_size=Config.IMAGE_SIZE
    )
    
    val_dataset = AnimalSegmentationDataset(
        Config.DATA_ROOT, 
        split='val',
        transform=transforms.val_transform,
        target_size=Config.IMAGE_SIZE
    )
    
    train_loader = DataLoader(train_dataset, 
                            batch_size=Config.BATCH_SIZE, 
                            shuffle=True)
    val_loader = DataLoader(val_dataset, 
                          batch_size=Config.BATCH_SIZE)
    
    # Model
    model = UNet(n_channels=3, n_classes=Config.NUM_CLASSES).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    
    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        config=Config
    )
    
    # Train
    trainer.train(Config.NUM_EPOCHS)

if __name__ == '__main__':
    main() 
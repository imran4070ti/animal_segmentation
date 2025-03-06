from pathlib import Path

class Config:
    # Dataset
    DATA_ROOT = Path('dataset/animal_segmentation')
    IMAGE_SIZE = (512, 512)
    NUM_CLASSES = 2
    
    # Training
    BATCH_SIZE = 8
    NUM_EPOCHS = 10
    LEARNING_RATE = 0.001
    DEVICE = 'cuda'
    
    # Checkpoints
    CHECKPOINT_DIR = Path('checkpoints')
    
    # Logging
    LOG_INTERVAL = 10 
    
    # Data Augmentation
    USE_AUGMENTATION = True  # Set to False to disable augmentation 
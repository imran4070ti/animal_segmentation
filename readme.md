# Animal Segmentation

A PyTorch-based implementation for semantic segmentation of animals using UNet architecture. This project provides a complete pipeline for training, evaluating, and performing inference on animal segmentation tasks.

## Features
- UNet architecture for semantic segmentation
- Data augmentation using Albumentations
- Training with checkpoint saving and resuming
- Comprehensive evaluation metrics
- Easy-to-use inference pipeline
- Support for custom datasets

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/animal_segmentation.git
cd animal_segmentation
```

2. Create and activate a virtual environment:
```bash
conda create -n animal_seg python=3.10
conda activate animal_seg
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Structure

The dataset should be organized as follows:
```
dataset/
├── train/
│   ├── images/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── labels/
│       ├── image1.txt
│       ├── image2.txt
│       └── ...
├── val/
│   ├── images/
│   │   └── ...
│   └── labels/
│       └── ...
├── test/
│   ├── images/
│   │   └── ...
│   └── labels/
│       └── ...
├── train_annotations.json
├── val_annotations.json
└── test_annotations.json
```

### Label Format
The labels are in YOLO format with polygon points:
```
<class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>
```

### Annotation Format
JSON files contain image information:
```json
{
    "images": [
        {
            "file_name": "image1.jpg",
            "width": 640,
            "height": 480
        },
        ...
    ]
}
```

## Usage

### Training

```python
from configs.config import Config
from models.unet import UNet
from preprocessing.dataset import AnimalSegmentationDataset
from preprocessing.transforms import Transforms
from utils.trainer import Trainer
import torch

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
        transform=transforms.train_transform,
        target_size=Config.IMAGE_SIZE,
        split='train'
    )
    
    val_dataset = AnimalSegmentationDataset(
        Config.DATA_ROOT, 
        transform=transforms.val_transform,
        target_size=Config.IMAGE_SIZE,
        split='val'
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=Config.BATCH_SIZE
    )
    
    # Model
    model = UNet(n_channels=3, n_classes=Config.NUM_CLASSES).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    
    # Train
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        config=Config
    )
    
    trainer.train(Config.NUM_EPOCHS)

if __name__ == '__main__':
    main()
```

Run training:
```bash
python train.py
```

### Evaluation

```python
from eval import evaluate

# Evaluate the model
metrics = evaluate('checkpoints/best_model.pth')
print(metrics)
```

### Inference

```python
from inference import inference
import cv2

# Run inference on a single image
image_path = 'test_images/sample.jpg'
checkpoint_path = 'checkpoints/best_model.pth'
prediction = inference(image_path, checkpoint_path)
```

## Configuration

You can modify the training configuration in `configs/config.py`:

```python
class Config:
    # Dataset
    DATA_ROOT = 'dataset/animal_segmentation'
    IMAGE_SIZE = (512, 512)
    NUM_CLASSES = 2
    
    # Training
    BATCH_SIZE = 8
    NUM_EPOCHS = 10
    LEARNING_RATE = 0.001
    DEVICE = 'cuda'
    
    # Data Augmentation
    USE_AUGMENTATION = True
```

## Results

The training process will generate:
- Model checkpoints in `checkpoints/`
- Training logs in `logs/`
- Loss curves and visualizations
- Evaluation metrics

## Monitoring Training

You can monitor the training progress:
1. Loss curves are saved in `checkpoints/loss_curves.png`
2. Logs are saved in `checkpoints/training.log`
3. Best model is saved as `checkpoints/best_model.pth`

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- UNet architecture implementation
- Albumentations for data augmentation
- PyTorch community

## Contact

Your Name - imranhasan.mhs13@gmail.com
Project Link: [https://github.com/imran4070ti/animal_segmentation](https://github.com/imran4070ti/animal_segmentation)

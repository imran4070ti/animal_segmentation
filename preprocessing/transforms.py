import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

class Transforms:
    def __init__(self, target_size=(256, 256), use_augmentation=False):
        """
        Args:
            target_size (tuple): Target size for resizing (height, width)
            use_augmentation (bool): Whether to use data augmentation for training
        """
        if use_augmentation:
            # Training transforms with augmentation
            self.train_transform = A.Compose([
                A.Resize(target_size[0], target_size[1]),
                A.OneOf([
                    A.RandomRotate90(p=0.5),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                ], p=0.5),
                A.OneOf([
                    A.GridDistortion(p=0.5),
                    A.OpticalDistortion(distort_limit=0.2, shift_limit=0.2, p=0.5),
                    A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
                ], p=0.3),
                A.OneOf([
                    A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                    A.RandomGamma(gamma_limit=(80, 120), p=0.5),
                ], p=0.3),
                A.OneOf([
                    A.Blur(blur_limit=3, p=0.5),
                    A.MedianBlur(blur_limit=3, p=0.5),
                ], p=0.2),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])
        else:
            # Training transforms without augmentation
            self.train_transform = A.Compose([
                A.Resize(target_size[0], target_size[1]),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])

        # Validation transforms (no augmentation)
        self.val_transform = A.Compose([
            A.Resize(target_size[0], target_size[1]),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])

    def train_transforms(self, image, mask):
        """Apply training transforms to image and mask"""
        transformed = self.train_transform(image=np.array(image), mask=np.array(mask))
        return transformed["image"], transformed["mask"]

    def val_transforms(self, image, mask):
        """Apply validation transforms to image and mask"""
        transformed = self.val_transform(image=np.array(image), mask=np.array(mask))
        return transformed["image"], transformed["mask"]

def preprocess_image(image, image_size):
    """
    Preprocess image for model inference
    Args:
        image: PIL Image
        image_size: tuple of (height, width)
    Returns:
        torch.Tensor: Preprocessed image tensor
    """
    # Convert PIL Image to numpy array
    image_np = np.array(image)
    
    # Define preprocessing transforms
    transform = A.Compose([
        A.Resize(height=image_size[0], width=image_size[1]),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2(),
    ])
    
    # Apply transforms
    transformed = transform(image=image_np)
    return transformed['image'] 
import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2
import albumentations as A

class AnimalSegmentationDataset(Dataset):
    def __init__(self, root_dir, transform=None, target_size=(256, 256), split='train'):
        """
        Args:
            root_dir (str): Root directory of the dataset
            transform (callable, optional): Optional transform to be applied on images
            target_size (tuple): Target size for resizing images and masks
            split (str): 'train', 'val', or 'test'
        """
        self.root_dir = root_dir
        self.transform = transform
        self.target_size = target_size
        self.split = split
        
        # Load annotations
        anno_file = os.path.join(root_dir, f'{split}_annotations.json')
        if not os.path.exists(anno_file):
            raise FileNotFoundError(f"Annotation file not found: {anno_file}")
            
        with open(anno_file, 'r') as f:
            self.annotations = json.load(f)
            
        self.images = self.annotations['images']
        self.image_dir = os.path.join(root_dir, split, 'images')
        self.label_dir = os.path.join(root_dir, split, 'labels')
        
        # Verify directories exist
        if not os.path.exists(self.image_dir):
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not os.path.exists(self.label_dir):
            raise FileNotFoundError(f"Label directory not found: {self.label_dir}")
    
    def __len__(self):
        return len(self.images)
    
    def create_mask(self, label_path, image_size):
        """
        Create binary mask from YOLO format label file
        
        Args:
            label_path (str): Path to label file
            image_size (tuple): Original image size (height, width)
            
        Returns:
            np.ndarray: Binary mask
        """
        mask = np.zeros(image_size, dtype=np.uint8)
        
        if os.path.exists(label_path):
            try:
                with open(label_path, 'r') as f:
                    data = f.read().strip().split()
                    if len(data) > 0:
                        # Convert YOLO format to mask
                        class_id = int(float(data[0]))
                        
                        # Extract polygon points
                        polygon_points = []
                        for i in range(5, len(data), 2):
                            if i + 1 < len(data):
                                x = float(data[i]) * image_size[1]  # scale x coordinate
                                y = float(data[i + 1]) * image_size[0]  # scale y coordinate
                                polygon_points.append([x, y])
                        
                        if polygon_points:
                            points = np.array(polygon_points, dtype=np.int32)
                            cv2.fillPoly(mask, [points], 1)
            except Exception as e:
                print(f"Error processing file {label_path}: {str(e)}")
        
        return mask
    
    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index
                
        Returns:
            tuple: (image, mask) where mask is the segmentation mask
        """
        # Load image
        img_info = self.images[idx]
        img_path = os.path.join(self.image_dir, img_info['file_name'])
        image = Image.open(img_path).convert('RGB')
        
        # Get original image size
        width = img_info['width']
        height = img_info['height']
        
        # Load label
        label_path = os.path.join(self.label_dir, 
                                os.path.splitext(img_info['file_name'])[0] + '.txt')
        
        # Create mask from label file
        mask = self.create_mask(label_path, (height, width))
        
        # Apply transforms
        if self.transform is not None:
            transformed = self.transform(image=np.array(image), mask=mask)
            image = transformed['image']  # Already a tensor from ToTensorV2
            mask = transformed['mask']
            
            # Ensure mask is of type Long (int64)
            mask = torch.tensor(mask, dtype=torch.long)
        else:
            # If no transform provided, convert to tensor
            image = torch.from_numpy(np.array(image)).float().permute(2, 0, 1) / 255.0
            mask = torch.tensor(mask, dtype=torch.long)  # Explicitly convert to Long
        
        return image, mask

    def get_image_info(self, idx):
        """Get image information"""
        return self.images[idx] 
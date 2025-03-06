import torch
import cv2
from PIL import Image
import numpy as np

from configs.config import Config
from models.unet import UNet
from preprocessing.transforms import preprocess_image
from utils.visualization import visualize_prediction

def inference(image_path, checkpoint_path):
    # Setup
    device = torch.device(Config.DEVICE)
    
    # Model
    model = UNet(n_channels=3, n_classes=Config.NUM_CLASSES).to(device)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    original_image = np.array(image)  # Keep original for visualization
    original_size = (original_image.shape[0], original_image.shape[1])  # (height, width)
    input_tensor = preprocess_image(image, Config.IMAGE_SIZE).to(device)
    
    # Inference
    with torch.no_grad():
        output = model(input_tensor.unsqueeze(0))
        pred = torch.argmax(output, dim=1).cpu().numpy()[0]
    
    # Resize prediction back to original image size
    pred_resized = cv2.resize(pred.astype(np.uint8), 
                            (original_size[1], original_size[0]),  # cv2.resize expects (width, height)
                            interpolation=cv2.INTER_NEAREST)  # Use nearest neighbor to preserve label values
    
    # Visualize
    visualize_prediction(original_image, pred_resized, save_path='prediction.png')
    
    return pred_resized

if __name__ == '__main__':
    image_path = 'dataset/animal_segmentation/test/0e9172e39fd3a8d3191d1e831804dd17.JPG'
    checkpoint_path = 'checkpoints/animal_seg_experiment/best_model.pth'
    prediction = inference(image_path, checkpoint_path) 
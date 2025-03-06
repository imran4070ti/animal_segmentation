import torch
from pathlib import Path
from tqdm import tqdm
import time
from datetime import datetime
import json
import numpy as np

from .logger import setup_logger
from .metrics import calculate_metrics
from .visualization import plot_losses

class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, 
                 optimizer, device, config):
        """
        Initialize the trainer
        
        Args:
            model: PyTorch model
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function
            optimizer: Optimizer
            device: Device to train on
            config: Configuration object
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.config = config
        
        # Setup logging
        self.logger = setup_logger(config.CHECKPOINT_DIR, 'training')
        
        # Create checkpoint directory
        self.checkpoint_dir = Path(config.CHECKPOINT_DIR)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize tracking variables
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.current_epoch = 0
        
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        epoch_metrics = {}
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch + 1}')
        for batch_idx, (images, masks) in enumerate(pbar):
            # Move data to device
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            running_loss += loss.item()
            current_loss = running_loss / (batch_idx + 1)
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{current_loss:.4f}'})
            
            # Log batch metrics
            if batch_idx % self.config.LOG_INTERVAL == 0:
                self.logger.info(
                    f'Epoch: {self.current_epoch + 1}, '
                    f'Batch: {batch_idx}/{len(self.train_loader)}, '
                    f'Loss: {loss.item():.4f}'
                )
        
        epoch_loss = running_loss / len(self.train_loader)
        return epoch_loss
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        metrics_sum = {}
        
        with torch.no_grad():
            for images, masks in tqdm(self.val_loader, desc='Validation'):
                # Move data to device
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                running_loss += loss.item()
                
                # Calculate additional metrics
                predictions = torch.argmax(outputs, dim=1)
                batch_metrics = calculate_metrics(predictions, masks)
                
                # Accumulate metrics
                for k, v in batch_metrics.items():
                    metrics_sum[k] = metrics_sum.get(k, 0) + v
        
        # Calculate average metrics
        val_loss = running_loss / len(self.val_loader)
        metrics = {k: v / len(self.val_loader) for k, v in metrics_sum.items()}
        metrics['loss'] = val_loss
        
        return metrics
    
    def save_checkpoint(self, filename, metrics):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'metrics': metrics,
            'config': self.config.__dict__
        }
        
        torch.save(checkpoint, self.checkpoint_dir / filename)
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.best_val_loss = checkpoint['best_val_loss']
        
        return checkpoint.get('metrics', {})
    
    def train(self, num_epochs, resume_from=None):
        """
        Train the model
        
        Args:
            num_epochs: Number of epochs to train
            resume_from: Path to checkpoint to resume from (optional)
        """
        start_time = time.time()
        
        # Resume from checkpoint if specified
        if resume_from is not None:
            self.logger.info(f"Resuming from checkpoint: {resume_from}")
            metrics = self.load_checkpoint(resume_from)
            self.logger.info(f"Resumed from epoch {self.current_epoch}")
        
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # Training phase
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validation phase
            val_metrics = self.validate()
            val_loss = val_metrics['loss']
            self.val_losses.append(val_loss)
            
            # Log metrics
            epoch_time = time.time() - epoch_start_time
            self.logger.info(
                f'Epoch {epoch + 1}/{num_epochs} - '
                f'Time: {epoch_time:.2f}s - '
                f'Train Loss: {train_loss:.4f} - '
                f'Val Loss: {val_loss:.4f}'
            )
            
            for k, v in val_metrics.items():
                if k != 'loss':
                    self.logger.info(f'Val {k}: {v:.4f}')
            
            # Save checkpoints
            self.save_checkpoint('latest_model.pth', val_metrics)
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint('best_model.pth', val_metrics)
                self.logger.info(
                    f'New best model saved! '
                    f'Validation loss: {self.best_val_loss:.4f}'
                )
            
            # Plot training history
            plot_losses(self.train_losses, self.val_losses, self.checkpoint_dir)
            
            # Save training stats
            stats = {
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'best_val_loss': self.best_val_loss,
                'last_epoch': self.current_epoch,
                'training_time': time.time() - start_time
            }
            
            with open(self.checkpoint_dir / 'training_stats.json', 'w') as f:
                json.dump(stats, f, indent=4)
        
        total_time = time.time() - start_time
        self.logger.info(f'Training completed! Total time: {total_time:.2f}s')
        
        return self.train_losses, self.val_losses, self.best_val_loss 
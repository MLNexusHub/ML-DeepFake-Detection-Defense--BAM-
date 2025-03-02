import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
from typing import Tuple, List
from tqdm import tqdm  # Add this at the top with other imports

class InvertibleNN(nn.Module):
    """
    Invertible Neural Network with encoder and decoder components.
    """
    def __init__(self, in_channels: int = 3, latent_dim: int = 16):
        super(InvertibleNN, self).__init__()
        
        # Encoder network
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, latent_dim, 3, padding=1)
        )
        
        # Decoder network
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_dim, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, in_channels, 3, padding=1),
            nn.Sigmoid()  # Ensure output is in [0,1]
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through both encoder and decoder.
        Returns both latent representation and reconstruction.
        """
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return latent, reconstruction
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent representation."""
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to image."""
        return self.decoder(z)

class ImageDataset(Dataset):
    """Dataset class for loading images."""
    def __init__(self, data_dir: str, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.image_paths = []
        
        # Recursively find all images
        valid_extensions = {'.jpg', '.jpeg', '.png'}
        for ext in valid_extensions:
            self.image_paths.extend(list(self.data_dir.rglob(f'*{ext}')))
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    device: torch.device,
    learning_rate: float = 1e-4,
    save_path: str = 'inn_model.pth'
) -> List[float]:
    """
    Train the invertible neural network.
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Loss functions
    reconstruction_criterion = nn.MSELoss()
    invertibility_criterion = nn.L1Loss()
    
    # Training history
    history = []
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        # Add progress bar for each epoch
        pbar = tqdm(train_loader, desc=f'Epoch [{epoch+1}/{num_epochs}]')
        for batch in pbar:
            # Move batch to device
            images = batch.to(device)
            
            # Forward pass
            latent, reconstructed = model(images)
            
            # Compute losses
            reconstruction_loss = reconstruction_criterion(reconstructed, images)
            
            # Invertibility loss: ensure decoder(encoder(x)) ≈ x
            decoded_again = model.decode(latent)
            invertibility_loss = invertibility_criterion(decoded_again, images)
            
            # Total loss
            loss = reconstruction_loss + invertibility_loss
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                images = batch.to(device)
                latent, reconstructed = model(images)
                val_loss += reconstruction_criterion(reconstructed, images).item()
        
        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        history.append(avg_val_loss)
        
        print(f'\nEpoch [{epoch+1}/{num_epochs}]')
        print(f'Training Loss: {avg_train_loss:.4f}')
        print(f'Validation Loss: {avg_val_loss:.4f}')
        
        # Save best model
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print(f'Model saved to {save_path}')
        
        print('-' * 50)
    
    return history

def main():
    # Check GPU availability
    if not torch.cuda.is_available():
        raise RuntimeError("This script requires a GPU to run. No GPU was detected.")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    
    # Force GPU usage
    device = torch.device('cuda')
    print(f'Using GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')
    
    # Data directories
    data_dir = 'D:/A3GIS/data'  # Your data directory
    
    # Image transformations
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize images to fixed size
        transforms.ToTensor(),          # Convert to tensor and scale to [0,1]
    ])
    
    # Create datasets
    full_dataset = ImageDataset(data_dir, transform=transform)
    print(f'Total number of images found: {len(full_dataset)}')
    
    # Split dataset
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    print(f'Training samples: {len(train_dataset)}')
    print(f'Validation samples: {len(val_dataset)}')
    
    # Create data loaders with pin_memory for faster GPU transfer
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize model
    model = InvertibleNN(in_channels=3, latent_dim=16)
    model = model.cuda()
    
    # Training parameters - REDUCED EPOCHS
    num_epochs = 5  # Changed from 50 to 5
    learning_rate = 1e-4
    
    try:
        # Train the model
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=num_epochs,
            device=device,
            learning_rate=learning_rate,
            save_path='inn_model.pth'
        )
        
        print('Training completed successfully!')
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            print('ERROR: GPU out of memory. Try reducing batch size or image dimensions.')
        else:
            print(f'ERROR: {str(e)}')
        raise e

if __name__ == '__main__':
    main() 
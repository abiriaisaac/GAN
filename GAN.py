# -*- coding: utf-8 -*-
"""
Fatigue Data Analysis and GAN-based Data Generation
Created on Thu Aug 15 07:59:36 2024
@author: Abiria_Isaac
"""

# ======================
# 1. IMPORT LIBRARIES
# ======================
import pandas as pd  # Data manipulation and analysis
import torch  # PyTorch deep learning framework
import torch.nn as nn  # Neural network modules
import torch.optim as optim  # Optimization algorithms
import matplotlib.pyplot as plt  # Data visualization
import seaborn as sns  # Enhanced visualization
import numpy as np  # Numerical operations
from matplotlib.gridspec import GridSpec  # Complex plot layouts

# ======================
# 2. VISUALIZATION SETUP
# ======================
# Set color palette and global plot parameters
sns.set_palette("colorblind")  # Colorblind-friendly palette
plt.rcParams.update({
    'font.size': 16,            # Base font size
    'axes.titlesize': 14,       # Title size
    'axes.labelsize': 16,       # Axis label size
    'xtick.labelsize': 16,      # X-tick label size
    'ytick.labelsize': 16,      # Y-tick label size
    'figure.dpi': 300,          # Figure resolution
    'savefig.dpi': 300,         # Saved image resolution
    'savefig.bbox': 'tight',    # Tight bounding box
    'figure.autolayout': True,  # Auto layout adjustment
    'font.weight': 'bold',      # Bold text
    'axes.titleweight': 'bold', # Bold titles
    'axes.labelweight': 'bold'  # Bold labels
})

# ======================
# 3. DATA LOADING AND ANALYSIS
# ======================
def load_and_analyze_data(file_path):
    """Load fatigue data and provide initial analysis"""
    data = pd.read_csv(file_path)  # Read CSV file
    
    print("\n=== Data Summary ===")
    print(data.describe())  # Show statistical summary
    return data

# ======================
# 4. GAN IMPLEMENTATION
# ======================
class Generator(nn.Module):
    """Generator network that creates synthetic data"""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),       # Input layer
            nn.LeakyReLU(0.2),              # Activation
            nn.Linear(128, 256),            # Hidden layer
            nn.BatchNorm1d(256),           # Normalization
            nn.LeakyReLU(0.2),              # Activation
            nn.Linear(256, output_dim),     # Output layer
            nn.Tanh()                       # Final activation
        )

    def forward(self, x):
        return self.model(x)  # Forward pass

class Discriminator(nn.Module):
    """Discriminator network that evaluates data authenticity"""
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),      # Input layer
            nn.LeakyReLU(0.2),              # Activation
            nn.Dropout(0.3),                # Regularization
            nn.Linear(256, 128),            # Hidden layer
            nn.LeakyReLU(0.2),              # Activation
            nn.Dropout(0.3),                # Regularization
            nn.Linear(128, 1),              # Output layer
            nn.Sigmoid()                    # Probability output
        )

    def forward(self, x):
        return self.model(x)  # Forward pass

def train_gan(data_tensor, input_dim, output_dim, num_epochs=1000, batch_size=64):
    """Train the GAN model"""
    # Initialize networks
    generator = Generator(input_dim, output_dim)
    discriminator = Discriminator(output_dim)
    
    # Set up optimizers
    optimizer_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    criterion = nn.BCELoss()  # Binary cross-entropy loss
    
    # Track training progress
    losses_g, losses_d = [], []
    
    # Training loop
    for epoch in range(num_epochs):
        for i in range(0, len(data_tensor), batch_size):
            # Prepare real data
            real_data = data_tensor[i:i+batch_size]
            real_labels = torch.ones(real_data.size(0), 1)
            
            # Generate fake data
            noise = torch.randn(real_data.size(0), input_dim)
            fake_data = generator(noise)
            fake_labels = torch.zeros(real_data.size(0), 1)
            
            # Train discriminator
            optimizer_d.zero_grad()
            loss_real = criterion(discriminator(real_data), real_labels)
            loss_fake = criterion(discriminator(fake_data.detach()), fake_labels)
            loss_d = (loss_real + loss_fake) / 2  # Average loss
            loss_d.backward()
            optimizer_d.step()
            
            # Train generator
            optimizer_g.zero_grad()
            output = discriminator(fake_data)
            loss_g = criterion(output, real_labels)
            loss_g.backward()
            optimizer_g.step()
        
        # Record losses
        losses_g.append(loss_g.item())
        losses_d.append(loss_d.item())
        
        # Print progress
        if epoch % 500 == 0:
            print(f"Epoch {epoch}: G Loss {loss_g.item():.4f}, D Loss {loss_d.item():.4f}")
    
    return generator, losses_g, losses_d

# ======================
# 5. VISUALIZATION FUNCTIONS
# ======================
def plot_training_history(losses_g, losses_d):
    """Plot generator and discriminator loss during training"""
    plt.figure(figsize=(10, 5))
    plt.plot(losses_g, label='Generator Loss')
    plt.plot(losses_d, label='Discriminator Loss')
    plt.title('GAN Training History', fontsize=14)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def compare_distributions(original, generated, columns):
    """Compare distributions of original and generated features"""
    fig, axes = plt.subplots(1, len(columns), figsize=(15, 4))
    for i, col in enumerate(columns):
        sns.kdeplot(original[col], ax=axes[i], label='Original', fill=True)
        sns.kdeplot(generated[col], ax=axes[i], label='Generated', fill=True)
        axes[i].set_title(f'{col} Distribution')
        axes[i].legend()
    plt.suptitle('Original vs Generated Distributions', fontsize=14, y=1.05)
    plt.tight_layout()
    plt.show()

def compare_correlations(original, generated):
    """Compare correlation matrices between original and generated data"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Original data correlation
    sns.heatmap(original.corr(), annot=True, cmap='coolwarm', 
                center=0, fmt=".2f", ax=ax1, cbar=False)
    ax1.set_title('Original Data Correlation')
    
    # Generated data correlation
    sns.heatmap(generated.corr(), annot=True, cmap='coolwarm', 
                center=0, fmt=".2f", ax=ax2)
    ax2.set_title('Generated Data Correlation')
    
    plt.tight_layout()
    plt.show()

# ======================
# 6. MAIN EXECUTION
# ======================
if __name__ == "__main__":
    # Load and analyze fatigue data
    file_path = 'Original_fatigue_data.csv'  # Combined dataset
    data = load_and_analyze_data(file_path)
    
    # Normalize data for GAN training
    data_normalized = (data - data.min()) / (data.max() - data.min())
    data_tensor = torch.tensor(data_normalized.values, dtype=torch.float32)
    
    # Train GAN model
    input_dim = 100  # Noise vector dimension
    output_dim = data.shape[1]  # Matches number of features
    generator, losses_g, losses_d = train_gan(data_tensor, input_dim, output_dim)
    
    # Visualize training progress
    plot_training_history(losses_g, losses_d)
    
    # Generate synthetic fatigue data
    num_samples = 60  # Number of synthetic samples
    noise = torch.randn(num_samples, input_dim)
    generated_data = generator(noise).detach().numpy()
    
    # Rescale to original data range
    generated_data = generated_data * (data.max().values - data.min().values) + data.min().values
    generated_df = pd.DataFrame(generated_data, columns=data.columns)
    
    # Combine and save datasets
    combined_data = pd.concat([data, generated_df], ignore_index=True)
    combined_data.to_csv('Generated_fatigue_data.csv', index=False)
    
    # Compare data characteristics
    compare_distributions(data, generated_df, data.columns)
    compare_correlations(data, generated_df)
    
    # Create scatter plot comparisons
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Stress vs Cycles comparison
    sns.scatterplot(data=data, x='stress', y='cycle', ax=ax1, 
                   label='Original', s=100, alpha=0.7, edgecolor='k')
    sns.scatterplot(data=generated_df, x='stress', y='cycle', ax=ax1, 
                   label='Generated', s=100, alpha=0.7, edgecolor='k')
    ax1.set_title('Stress Amplitude vs Cycle')
    
    # Stress vs Defect Size comparison
    sns.scatterplot(data=data, x='stress', y='defect_size', ax=ax2, 
                   label='Original', s=100, alpha=0.7, edgecolor='k')
    sns.scatterplot(data=generated_df, x='stress', y='defect_size', ax=ax2, 
                   label='Generated', s=100, alpha=0.7, edgecolor='k')
    ax2.set_title('Stress Amplitude vs Defect Size')
    
    plt.tight_layout()
    plt.show()

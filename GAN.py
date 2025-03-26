

# -*- coding: utf-8 -*-
"""
Fatigue Data Analysis and GAN-based Data Generation
Created on Thu Aug 15 07:59:36 2024
@author: Abiria_Isaac


"""

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.gridspec import GridSpec

# ======================
# 1. SETUP AND CONFIG
# ======================
sns.set_palette("colorblind")
plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 14,
    'axes.labelsize': 16,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'figure.autolayout': True,
    'font.weight': 'bold',  # Make all text bold
    'axes.titleweight': 'bold',  # Bold axes titles
    'axes.labelweight': 'bold'  # Bold axes labels
})

# ======================
# 2. DATA LOADING AND EDA
# ======================
def load_and_analyze_data(file_path):
    """Load data and perform initial analysis"""
    data = pd.read_csv(file_path)
    
    print("\n=== Data Summary ===")
    print(data.describe())
    return data

# ======================
# 3. GAN IMPLEMENTATION
# ======================
class Generator(nn.Module):
    """Generator network for GAN"""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, output_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    """Discriminator network for GAN"""
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

def train_gan(data_tensor, input_dim, output_dim, num_epochs=1000, batch_size=64):
    """Train GAN model"""
    # Initialize models
    generator = Generator(input_dim, output_dim)
    discriminator = Discriminator(output_dim)
    
    # Optimizers
    optimizer_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    criterion = nn.BCELoss()
    
    # Training loop
    losses_g, losses_d = [], []
    for epoch in range(num_epochs):
        for i in range(0, len(data_tensor), batch_size):
            # Train discriminator
            real_data = data_tensor[i:i+batch_size]
            real_labels = torch.ones(real_data.size(0), 1)
            
            noise = torch.randn(real_data.size(0), input_dim)
            fake_data = generator(noise)
            fake_labels = torch.zeros(real_data.size(0), 1)
            
            optimizer_d.zero_grad()
            loss_real = criterion(discriminator(real_data), real_labels)
            loss_fake = criterion(discriminator(fake_data.detach()), fake_labels)
            loss_d = (loss_real + loss_fake) / 2
            loss_d.backward()
            optimizer_d.step()
            
            # Train generator
            optimizer_g.zero_grad()
            output = discriminator(fake_data)
            loss_g = criterion(output, real_labels)
            loss_g.backward()
            optimizer_g.step()
        
        losses_g.append(loss_g.item())
        losses_d.append(loss_d.item())
        
        if epoch % 500 == 0:
            print(f"Epoch {epoch}: G Loss {loss_g.item():.4f}, D Loss {loss_d.item():.4f}")
    
    return generator, losses_g, losses_d

# ======================
# 4. VISUALIZATION
# ======================
def plot_training_history(losses_g, losses_d):
    """Plot GAN training history"""
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
    """Compare original and generated distributions"""
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
    """Compare correlation matrices"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    sns.heatmap(original.corr(), annot=True, cmap='coolwarm', 
                center=0, fmt=".2f", ax=ax1, cbar=False)
    ax1.set_title('Original Data Correlation')
    
    sns.heatmap(generated.corr(), annot=True, cmap='coolwarm', 
                center=0, fmt=".2f", ax=ax2)
    ax2.set_title('Generated Data Correlation')
    
   
    plt.tight_layout()
    plt.show()

# ======================
# 5. MAIN EXECUTION
# ======================
if __name__ == "__main__":
    # Load and analyze data   Replace with the actual file#
    #file_path = 'Fu_90.csv'
    #file_path = 'Li_0.csv'
    #file_path = 'Li_90.csv'
    file_path = 'Original_fatigue_data.csv' #This is the combined file containing 'Fu_90.csv','Li_0.csv','Li_90.csv' 
    data = load_and_analyze_data(file_path)
    
    # Normalize data
    data_normalized = (data - data.min()) / (data.max() - data.min())
    data_tensor = torch.tensor(data_normalized.values, dtype=torch.float32)
    
    # Train GAN
    input_dim = 100
    output_dim = data.shape[1]
    generator, losses_g, losses_d = train_gan(data_tensor, input_dim, output_dim)
    
    # Plot training history
    plot_training_history(losses_g, losses_d)
    
    # Generate synthetic data
    num_samples = 60 #Adjust number accordingly
    noise = torch.randn(num_samples, input_dim)
    generated_data = generator(noise).detach().numpy()
    generated_data = generated_data * (data.max().values - data.min().values) + data.min().values
    generated_df = pd.DataFrame(generated_data, columns=data.columns)
    
    # Combine and save data
    combined_data = pd.concat([data, generated_df], ignore_index=True)
    combined_data.to_csv('Generated_fatigue_data.csv', index=False)
    
    # Visual comparisons
    compare_distributions(data, generated_df, data.columns)
    compare_correlations(data, generated_df)
    
    # Scatter plot comparisons
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    sns.scatterplot(data=data, x='stress', y='cycle', ax=ax1, label='Original', s=100, alpha=0.7, edgecolor='k')
    sns.scatterplot(data=generated_df, x='stress', y='cycle', ax=ax1, label='Generated', s=100, alpha=0.7, edgecolor='k')
    ax1.set_title('Stress Amplitude vs Cycle')
    
    sns.scatterplot(data=data, x='stress', y='defect_size', ax=ax2, label='Original', s=100, alpha=0.7, edgecolor='k')
    sns.scatterplot(data=generated_df, x='stress', y='defect_size', ax=ax2, label='Generated', s=100, alpha=0.7, edgecolor='k')
    ax2.set_title('Stress Amplitude vs Defect Size')
    
    
    plt.tight_layout()
    plt.show()
    
    
    
    
    

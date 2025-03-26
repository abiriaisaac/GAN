# Fatigue Data Analysis and GAN-based Data Generation

![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.8%2B-orange)
![Pandas](https://img.shields.io/badge/Pandas-1.2%2B-brightgreen)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.4%2B-blueviolet)

This project implements a Generative Adversarial Network (GAN) to generate synthetic fatigue data while preserving the statistical properties of the original dataset. The solution includes comprehensive exploratory data analysis, GAN implementation, and visualization tools to compare original and generated data distributions.

## Features

- **Data Loading & Analysis**: Automated EDA with correlation matrices and key relationship visualizations
- **GAN Implementation**: Custom generator and discriminator networks with configurable architecture
- **Training Pipeline**: Complete the training loop with adjustable hyperparameters
- **Visualization Tools**: 
  - Training loss history
  - Distribution comparisons
  - Correlation matrix comparisons
  - Scatter plot validations
- **Data Export**: Combined original and generated data export functionality

## Installation
1.Install the necessary libraries
2. Clone this repository:
   ```bash
   git clone https://github.com/abiriaisaac/Generative-Adversarial-Network.git
   cd fatigue-data-gan

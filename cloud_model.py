import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import rasterio
from glob import glob
import json
from pathlib import Path
import logging
import sys
import warnings
import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union, Any
import tempfile

import wandb
from prettytable import PrettyTable
from tqdm import tqdm
import albumentations as A
import logging
import sys
from datetime import datetime
from pathlib import Path
import warnings
import wandb
from prettytable import PrettyTable
from tqdm import tqdm
import albumentations as A
from torch.utils.tensorboard import SummaryWriter
import tempfile
from typing import Dict, List, Optional, Tuple, Union
import math
import torch.nn.functional as F
from collections import defaultdict
from typing import Any

# --- Configuration ---
DATA_PATH = 'data/'
METADATA_FILE = os.path.join(DATA_PATH, 'image_metadata.csv')
MODEL_SAVE_PATH = 'models_pytorch/'
RESULTS_PATH = 'results_pytorch/'
PLOTS_PATH = os.path.join(RESULTS_PATH, 'plots')
METRICS_PATH = os.path.join(RESULTS_PATH, 'metrics')

# Model parameters
IMG_HEIGHT = 128
IMG_WIDTH = 128
N_PAST_TIMESTEPS = 16  # 4 hours of data, assuming 15-min intervals
N_FUTURE_TIMESTEPS = [1, 2, 3]  # 15, 30, 45 mins prediction
EPOCHS = 100  # Increased for thorough training
BATCH_SIZE = 64  # Increased for faster training
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 15
LR_PATIENCE = 5
TRAIN_VAL_TEST_SPLIT = [0.7, 0.15, 0.15]  # Train/Val/Test split ratios

# Create all necessary directories
for path in [MODEL_SAVE_PATH, RESULTS_PATH, PLOTS_PATH, METRICS_PATH]:
    os.makedirs(path, exist_ok=True)

# --- Device Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- Create directories if they don't exist ---
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
os.makedirs(RESULTS_PATH, exist_ok=True)
# os.makedirs(IMAGE_DIR, exist_ok=True) # IMAGE_DIR is less directly used now

# --- Helper Functions ---

def load_and_preprocess_tif_image(folder_path_str):
    """
    Loads and preprocesses TIF images from a given folder path.
    Placeholder: Reads the first band of the first .tif file found.
    Output is HWC (Height, Width, Channels=1).
    """
    try:
        # Construct full path relative to DATA_PATH
        full_folder_path = os.path.join(DATA_PATH, folder_path_str)
        
        tif_files = glob(os.path.join(full_folder_path, '*.tif')) + glob(os.path.join(full_folder_path, '*.tiff'))
        
        if not tif_files:
            print(f"Warning: No .tif files found in {full_folder_path}. Skipping.")
            return None
            
        # Placeholder: use the first .tif file found
        image_path = tif_files[0]
        print(f"Processing TIF: {image_path}")

        with rasterio.open(image_path) as src:
            # Placeholder: read the first band.
            # You might need to select specific bands or combine them.
            img_array = src.read(1).astype(np.float32) # Read first band

            # Handle NoData values if necessary (e.g., fill with 0 or mean)
            if src.nodata is not None:
                img_array[img_array == src.nodata] = 0 # Example: fill nodata with 0

        if img_array is None:
            print(f"Warning: Could not read image data from {image_path}. Skipping.")
            return None

        # Resize
        img_resized = cv2.resize(img_array, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_LINEAR)
        
        # Normalize: TIF files can have various ranges.
        # This simple normalization might need adjustment based on your data's characteristics.
        min_val, max_val = np.min(img_resized), np.max(img_resized)
        if max_val > min_val:
            img_normalized = (img_resized - min_val) / (max_val - min_val)
        else:
            img_normalized = np.zeros_like(img_resized) # Avoid division by zero if flat image
            
        img_normalized = np.expand_dims(img_normalized, axis=-1) # Add channel dimension -> (H, W, 1)
        return img_normalized.astype(np.float32)

    except Exception as e:
        print(f"Error processing TIF folder {folder_path_str}: {e}")
        return None

def calculate_cloud_coverage_from_image(image_array):
    """
    Calculates cloud coverage percentage from a preprocessed image array.
    image_array is expected to be (H, W, C) with values normalized to [0,1].
    
    The function uses a combination of thresholding techniques to identify clouds:
    1. Brightness threshold: Clouds are typically brighter than other features
    2. Local contrast: Clouds often have higher local contrast
    3. Texture analysis: Using local standard deviation
    """
    if image_array is None or image_array.size == 0:
        print("Warning: Empty or None image array provided")
        return 0.0
    
    # Remove channel dimension if present
    if len(image_array.shape) == 3:
        image_array = image_array[..., 0]
    
    try:
        # 1. Brightness-based cloud detection
        brightness_threshold = 0.75  # Assuming normalized [0,1] values
        bright_pixels = image_array > brightness_threshold
        
        # 2. Local contrast analysis
        kernel_size = 3
        kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
        local_mean = cv2.filter2D(image_array, -1, kernel)
        local_contrast = np.abs(image_array - local_mean)
        contrast_threshold = 0.1  # Adjust based on your data
        high_contrast = local_contrast > contrast_threshold
        
        # 3. Texture analysis using local standard deviation
        local_std = np.zeros_like(image_array)
        padding = kernel_size // 2
        padded_img = np.pad(image_array, padding, mode='reflect')
        for i in range(padding, padded_img.shape[0] - padding):
            for j in range(padding, padded_img.shape[1] - padding):
                window = padded_img[i-padding:i+padding+1, j-padding:j+padding+1]
                local_std[i-padding, j-padding] = np.std(window)
        texture_threshold = 0.05  # Adjust based on your data
        textured_regions = local_std > texture_threshold
        
        # Combine all features
        cloud_pixels = bright_pixels & (high_contrast | textured_regions)
        
        # Calculate coverage percentage
        total_pixels = image_array.size
        cloudy_pixels = np.sum(cloud_pixels)
        coverage = (cloudy_pixels / total_pixels) * 100
        
        return float(coverage)
    
    except Exception as e:
        print(f"Error calculating cloud coverage: {e}")
        return 0.0

def load_and_preprocess_image(image_path):
    """Loads and preprocesses a single image. Output is HWC (Height, Width, Channels)."""
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) # Load as grayscale
        if img is None:
            print(f"Warning: Could not read image {image_path}. Skipping.")
            return None
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        img = img / 255.0 # Normalize to [0, 1]
        img = np.expand_dims(img, axis=-1) # Add channel dimension -> (H, W, 1)
        return img.astype(np.float32)
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def create_sequences(image_data, coverage_data, n_past, n_future_steps_list):
    """
    Creates sequences of past images and future coverage values.
    Images are expected to be (H, W, C). Output X will be (num_samples, n_past, H, W, C).
    """
    X, Y = {}, {}
    for n_future_steps in n_future_steps_list:
        X[n_future_steps] = []
        Y[n_future_steps] = []

    for i in range(len(image_data) - n_past - max(n_future_steps_list) + 1):
        past_seq_images = image_data[i : i + n_past]
        if any(x is None for x in past_seq_images): # Skip sequences with missing images
            continue
        for n_future_steps in n_future_steps_list:
            future_val = coverage_data[i + n_past + n_future_steps - 1]
            X[n_future_steps].append(np.array(past_seq_images)) # Shape: (n_past, H, W, C)
            Y[n_future_steps].append(future_val)

    for n_future_steps in n_future_steps_list:
        if X[n_future_steps]: # Ensure there is data before converting to numpy array
            X[n_future_steps] = np.array(X[n_future_steps], dtype=np.float32)
            Y[n_future_steps] = np.array(Y[n_future_steps], dtype=np.float32).reshape(-1, 1)
        else: # Handle cases where no sequences were created (e.g. not enough data)
            X[n_future_steps] = np.empty((0, n_past, IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.float32)
            Y[n_future_steps] = np.empty((0, 1), dtype=np.float32)
    return X, Y

# --- PyTorch Model Definitions ---

class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding='same')
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding='same')
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        # Calculate the flattened size dynamically based on IMG_HEIGHT, IMG_WIDTH
        # After two 2x2 poolings, H becomes H/4, W becomes W/4
        self._flattened_size = (IMG_HEIGHT // 4) * (IMG_WIDTH // 4) * 64

    def forward(self, x): # x shape: (batch*timesteps, C, H, W)
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        return x

    @property
    def flattened_size(self):
        return self._flattened_size

class LSTMModel(nn.Module):
    def __init__(self, cnn_output_size, lstm_hidden_size=128, lstm_layers=2, dense_hidden_size=64, n_future=1):
        super().__init__()
        self.cnn = CNNFeatureExtractor()
        self.lstm1 = nn.LSTM(input_size=cnn_output_size, hidden_size=lstm_hidden_size, num_layers=1, batch_first=True, bidirectional=False)
        self.lstm2 = nn.LSTM(input_size=lstm_hidden_size, hidden_size=lstm_hidden_size//2, num_layers=1, batch_first=True, bidirectional=False)
        self.fc1 = nn.Linear(lstm_hidden_size//2, dense_hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(dense_hidden_size, n_future)

    def forward(self, x): # x shape: (batch, timesteps, H, W, C)
        batch_size, timesteps, H, W, C_in = x.size()
        x = x.permute(0, 1, 4, 2, 3) # (batch, timesteps, C, H, W)
        c_in = x.view(batch_size * timesteps, C_in, H, W)
        c_out = self.cnn(c_in)
        r_in = c_out.view(batch_size, timesteps, -1)
        
        lstm_out1, _ = self.lstm1(r_in)
        lstm_out2, _ = self.lstm2(lstm_out1) # lstm_out2 is (batch, timesteps, lstm_hidden_size//2)
        
        # We take the output of the last LSTM cell from the second LSTM layer
        last_lstm_out = lstm_out2[:, -1, :]
        
        out = self.relu(self.fc1(last_lstm_out))
        out = self.fc2(out)
        return out

class GRUModel(nn.Module):
    def __init__(self, cnn_output_size, gru_hidden_size=128, gru_layers=2, dense_hidden_size=64, n_future=1):
        super().__init__()
        self.cnn = CNNFeatureExtractor()
        self.gru1 = nn.GRU(input_size=cnn_output_size, hidden_size=gru_hidden_size, num_layers=1, batch_first=True)
        self.gru2 = nn.GRU(input_size=gru_hidden_size, hidden_size=gru_hidden_size//2, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(gru_hidden_size//2, dense_hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(dense_hidden_size, n_future)

    def forward(self, x): # x shape: (batch, timesteps, H, W, C)
        batch_size, timesteps, H, W, C_in = x.size()
        x = x.permute(0, 1, 4, 2, 3) # (batch, timesteps, C, H, W)
        c_in = x.view(batch_size * timesteps, C_in, H, W)
        c_out = self.cnn(c_in)
        r_in = c_out.view(batch_size, timesteps, -1)
        
        gru_out1, _ = self.gru1(r_in)
        gru_out2, _ = self.gru2(gru_out1)
        
        last_gru_out = gru_out2[:, -1, :]
        
        out = self.relu(self.fc1(last_gru_out))
        out = self.fc2(out)
        return out

class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout_rate, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout1(attn_output))
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_output))
        return x

class TransformerModel(nn.Module):
    def __init__(self, cnn_output_size, n_future=1, num_transformer_blocks=2, embed_dim=64, num_heads=4, ff_dim=64, dropout_rate=0.1):
        super().__init__()
        self.cnn = CNNFeatureExtractor()
        self.input_projection = nn.Linear(cnn_output_size, embed_dim)
        
        self.transformer_blocks = nn.ModuleList(
            [TransformerEncoderBlock(embed_dim, num_heads, ff_dim, dropout_rate) for _ in range(num_transformer_blocks)]
        )
        
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1) # To pool across timesteps
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(embed_dim, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, n_future)

    def forward(self, x): # x shape: (batch, timesteps, H, W, C)
        batch_size, timesteps, H, W, C_in = x.size()
        x = x.permute(0, 1, 4, 2, 3) # (batch, timesteps, C, H, W)
        c_in = x.view(batch_size * timesteps, C_in, H, W)
        c_out = self.cnn(c_in) # (batch*timesteps, cnn_output_size)
        
        # Project CNN features to embed_dim
        r_in = c_out.view(batch_size, timesteps, -1) # (batch, timesteps, cnn_output_size)
        projected_features = self.input_projection(r_in) # (batch, timesteps, embed_dim)

        transformer_out = projected_features
        for block in self.transformer_blocks:
            transformer_out = block(transformer_out) # (batch, timesteps, embed_dim)
        
        # Pool across the sequence dimension (timesteps)
        # For AdaptiveAvgPool1d, input should be (batch, embed_dim, timesteps)
        pooled_out = self.global_avg_pool(transformer_out.permute(0, 2, 1)).squeeze(-1) # (batch, embed_dim)
        
        out = self.dropout(pooled_out)
        out = self.relu(self.fc1(out))
        out = self.dropout(out) # Added dropout based on Keras version
        out = self.fc2(out)
        return out

class ResidualCNNLSTMModel(nn.Module):
    def __init__(self, cnn_output_size, lstm_hidden_size=64, dense_hidden_size=64, n_future=1): # Reduced LSTM hidden size
        super().__init__()
        self.cnn = CNNFeatureExtractor()
        # For residual connection, ensure LSTM output dim matches a projection of CNN output
        self.lstm = nn.LSTM(input_size=cnn_output_size, hidden_size=lstm_hidden_size, num_layers=1, batch_first=True)
        
        # Projection for residual connection (if needed, or ensure dimensions match)
        # Here, we'll take the last CNN output and project it to match LSTM output
        self.residual_projection = nn.Linear(cnn_output_size, lstm_hidden_size) 
        
        self.fc1 = nn.Linear(lstm_hidden_size, dense_hidden_size) # Keras model had LSTM(64) -> Dense(64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(dense_hidden_size, n_future)

    def forward(self, x): # x shape: (batch, timesteps, H, W, C)
        batch_size, timesteps, H, W, C_in = x.size()
        x_permuted = x.permute(0, 1, 4, 2, 3) # (batch, timesteps, C, H, W)
        
        cnn_features_list = []
        for t in range(timesteps):
            # Process each timestep image through CNN
            # Input to CNN: (batch, C, H, W)
            cnn_feature_t = self.cnn(x_permuted[:, t, :, :, :]) # (batch, cnn_output_size)
            cnn_features_list.append(cnn_feature_t)
        
        # Stack features along time dimension for LSTM
        cnn_sequence_out = torch.stack(cnn_features_list, dim=1) # (batch, timesteps, cnn_output_size)
        
        lstm_out, _ = self.lstm(cnn_sequence_out) # (batch, timesteps, lstm_hidden_size)
        
        # Use the output of the last LSTM cell
        last_lstm_out = lstm_out[:, -1, :] # (batch, lstm_hidden_size)
        
        # Conceptual residual: take features of the *last input image* from CNN, project and add
        # This is a simplified interpretation of the Keras model's comment
        last_cnn_feature = cnn_features_list[-1] # (batch, cnn_output_size)
        projected_residual = self.residual_projection(last_cnn_feature) # (batch, lstm_hidden_size)
        
        # Add residual (element-wise)
        combined_out = last_lstm_out + projected_residual # (batch, lstm_hidden_size)
        
        out = self.relu(self.fc1(combined_out)) # Dense layer on top
        out = self.fc2(out)
        return out

class ImprovedCNNFeatureExtractor(nn.Module):
    """Improved CNN feature extractor with residual connections and batch normalization."""
    def __init__(self, config: CloudCoverageConfig):
        super().__init__()
        self.config = config
        
        def conv_block(in_channels, out_channels, stride=1):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=config.dropout_rate)
            )
        
        # Initial convolution
        self.conv1 = conv_block(1, config.cnn_channels[0])
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            nn.Sequential(
                conv_block(channels, channels),
                conv_block(channels, channels)
            ) for channels in config.cnn_channels
        ])
        
        # Transition blocks (downsampling)
        self.transitions = nn.ModuleList([
            conv_block(config.cnn_channels[i], config.cnn_channels[i+1], stride=2)
            for i in range(len(config.cnn_channels)-1)
        ])
        
        # Calculate output size
        self._calculate_output_size()
    
    def _calculate_output_size(self):
        """Calculate the output size of the CNN."""
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, self.config.img_height, self.config.img_width)
            output = self.forward(dummy_input)
            self._output_size = output.view(1, -1).shape[1]
    
    @property
    def output_size(self):
        return self._output_size
    
    def forward(self, x):
        x = self.conv1(x)
        
        for res_block, transition in zip(self.res_blocks[:-1], self.transitions):
            identity = x
            x = res_block(x)
            x = x + identity  # Residual connection
            x = transition(x)
        
        # Last residual block without transition
        identity = x
        x = self.res_blocks[-1](x)
        x = x + identity
        
        return x.flatten(start_dim=1)  # Flatten for the LSTM

class ImprovedLSTMModel(nn.Module):
    """Improved LSTM model with attention mechanism and skip connections."""
    def __init__(self, config: CloudCoverageConfig):
        super().__init__()
        self.config = config
        self.cnn = ImprovedCNNFeatureExtractor(config)
        cnn_output_size = self.cnn.output_size
        
        # Bidirectional LSTM layers
        self.lstm_layers = nn.ModuleList([
            nn.LSTM(
                input_size=cnn_output_size if i == 0 else config.lstm_hidden_size * 2,
                hidden_size=config.lstm_hidden_size,
                bidirectional=True,
                batch_first=True
            ) for i in range(2)  # 2-layer LSTM
        ])
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(config.lstm_hidden_size * 2, config.lstm_hidden_size),
            nn.Tanh(),
            nn.Linear(config.lstm_hidden_size, 1)
        )
        
        # Output layers
        self.fc1 = nn.Linear(config.lstm_hidden_size * 2, config.lstm_hidden_size)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.fc2 = nn.Linear(config.lstm_hidden_size, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.kaiming_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def attention_forward(self, lstm_output):
        """Apply attention mechanism to LSTM output."""
        attention_weights = self.attention(lstm_output)
        attention_weights = torch.softmax(attention_weights, dim=1)
        attended_out = torch.sum(attention_weights * lstm_output, dim=1)
        return attended_out, attention_weights
    
    def forward(self, x):
        batch_size, timesteps, H, W, C = x.size()
        
        # Process each timestep through CNN
        x = x.view(batch_size * timesteps, C, H, W)
        x = self.cnn(x)
        x = x.view(batch_size, timesteps, -1)
        
        # Process through LSTM layers with skip connections
        lstm_out = x
        all_states = []
        for lstm_layer in self.lstm_layers:
            new_states, _ = lstm_layer(lstm_out)
            if len(all_states) > 0:  # Add skip connection
                lstm_out = new_states + all_states[-1]
            else:
                lstm_out = new_states
            all_states.append(lstm_out)
        
        # Apply attention
        context, attention_weights = self.attention_forward(lstm_out)
        
        # Final prediction
        out = self.fc1(context)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out

class ImprovedTransformerModel(nn.Module):
    """Improved Transformer model with advanced features."""
    def __init__(self, config: CloudCoverageConfig):
        super().__init__()
        self.config = config
        self.cnn = ImprovedCNNFeatureExtractor(config)
        cnn_output_size = self.cnn.output_size
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(
            d_model=config.lstm_hidden_size,
            max_len=config.n_past_timesteps
        )
        
        # Input projection
        self.input_projection = nn.Linear(cnn_output_size, config.lstm_hidden_size)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.lstm_hidden_size,
            nhead=8,
            dim_feedforward=config.lstm_hidden_size * 4,
            dropout=config.dropout_rate,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        # Output layers
        self.fc1 = nn.Linear(config.lstm_hidden_size, config.lstm_hidden_size // 2)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.fc2 = nn.Linear(config.lstm_hidden_size // 2, 1)
    
    def forward(self, x):
        batch_size, timesteps, H, W, C = x.size()
        
        # Process through CNN
        x = x.view(batch_size * timesteps, C, H, W)
        x = self.cnn(x)
        x = x.view(batch_size, timesteps, -1)
        
        # Project to transformer dimension
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transform
        transformer_out = self.transformer_encoder(x)
        
        # Use the final timestep for prediction
        out = transformer_out[:, -1, :]
        
        # Final prediction
        out = self.fc1(out)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer model."""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:x.size(1)]
        return x

# --- Training and Evaluation Functions ---

def train_and_evaluate(model, X_data, y_data, model_name, future_interval, scaler_y):
    """Trains the PyTorch model with proper validation and comprehensive metrics."""
    print(f"--- Training {model_name} for {future_interval*15}-minute prediction (PyTorch) ---")
    start_time = datetime.now()
    
    # Split data into train, validation, and test sets
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X_data, y_data)
    
    # Move model to device and setup training
    model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=LR_PATIENCE, verbose=True)
    
    # Create dataloaders
    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float()),
                            batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float()),
                          batch_size=BATCH_SIZE)
    test_loader = DataLoader(TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float()),
                           batch_size=BATCH_SIZE)
    
    # Training history
    best_val_loss = float('inf')
    early_stop_counter = 0
    train_losses = []
    val_losses = []
    train_r2_scores = []
    val_r2_scores = []
    
    model_path = os.path.join(MODEL_SAVE_PATH, f'{model_name}_future_{future_interval*15}min.pth')
    
    print("Starting training...")
    for epoch in range(EPOCHS):
        # Training phase
        model.train()
        epoch_train_loss = 0
        train_preds = []
        train_targets = []
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item() * batch_X.size(0)
            train_preds.extend(outputs.detach().cpu().numpy())
            train_targets.extend(batch_y.cpu().numpy())
        
        # Evaluation phase
        train_metrics = evaluate_model(model, train_loader, criterion, DEVICE, scaler_y)
        val_metrics = evaluate_model(model, val_loader, criterion, DEVICE, scaler_y)
        
        # Update learning rate
        scheduler.step(val_metrics['loss'])
        
        # Save metrics
        train_losses.append(train_metrics['loss'])
        val_losses.append(val_metrics['loss'])
        train_r2_scores.append(train_metrics['r2'])
        val_r2_scores.append(val_metrics['r2'])
        
        # Print progress
        print(f"Epoch [{epoch+1}/{EPOCHS}]")
        print(f"Train Loss: {train_metrics['loss']:.4f}, Train R²: {train_metrics['r2']:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f}, Val R²: {val_metrics['r2']:.4f}")
        
        # Model checkpoint
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            early_stop_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_metrics['loss'],
                'val_r2': val_metrics['r2']
            }, model_path)
            print("Saved new best model!")
        else:
            early_stop_counter += 1
            if early_stop_counter >= EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered!")
                break
    
    # Plot learning curves
    plot_learning_curves(train_losses, val_losses, train_r2_scores, val_r2_scores, model_name, future_interval)
    
    # Load best model for final evaluation
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Final evaluation on test set
    test_metrics = evaluate_model(model, test_loader, criterion, DEVICE, scaler_y)
    
    # Create prediction vs actual plots
    plot_prediction_comparison(test_metrics['targets'], test_metrics['predictions'], 
                             model_name, future_interval)
    
    # Save final metrics
    training_duration = (datetime.now() - start_time).total_seconds() / 3600
    final_metrics = {
        'model_name': model_name,
        'prediction_horizon': f"{future_interval*15} minutes",
        'performance_metrics': {
            'test_loss': float(test_metrics['loss']),
            'test_r2': float(test_metrics['r2']),
            'test_rmse': float(test_metrics['rmse']),
            'test_mae': float(test_metrics['mae'])
        },
        'training_info': {
            'duration_hours': float(training_duration),
            'epochs_completed': epoch + 1,
            'early_stopped': early_stop_counter >= EARLY_STOPPING_PATIENCE,
            'final_learning_rate': float(optimizer.param_groups[0]['lr']),
            'best_validation_loss': float(best_val_loss)
        },
        'dataset_info': {
            'train_samples': len(train_loader.dataset),
            'validation_samples': len(val_loader.dataset),
            'test_samples': len(test_loader.dataset)
        },
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    save_model_metrics(final_metrics, model_name, future_interval)
    
    print("\nFinal Test Results:")
    print(f"Loss: {test_metrics['loss']:.4f}")
    print(f"R² Score: {test_metrics['r2']:.4f}")
    print(f"RMSE: {test_metrics['rmse']:.4f}")
    print(f"MAE: {test_metrics['mae']:.4f}")
    print(f"Training Duration: {training_duration:.2f} hours")
    
    return final_metrics

def plot_training_history(history, model_name, future_interval):
    """
    Plots training history including loss curves and metrics.
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.figure(figsize=(15, 5))
    
    # Plot loss curves
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title(f'Loss Curves - {model_name} ({future_interval*15}min)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot metrics
    plt.subplot(1, 2, 2)
    plt.plot(history['train_r2'], label='Training R²')
    plt.plot(history['val_r2'], label='Validation R²')
    plt.title(f'R² Score - {model_name} ({future_interval*15}min)')
    plt.xlabel('Epoch')
    plt.ylabel('R² Score')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(PLOTS_PATH, f'training_history_{model_name}_{future_interval*15}min_{timestamp}.png')
    plt.savefig(plot_path)
    plt.close()

def plot_prediction_comparison(y_true, y_pred, model_name, future_interval):
    """
    Creates scatter and residual plots for model predictions.
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.figure(figsize=(15, 5))
    
    # Scatter plot
    plt.subplot(1, 2, 1)
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.title(f'Predicted vs Actual - {model_name} ({future_interval*15}min)')
    plt.xlabel('Actual Cloud Coverage (%)')
    plt.ylabel('Predicted Cloud Coverage (%)')
    plt.grid(True)
    
    # Residual plot
    residuals = y_pred - y_true
    plt.subplot(1, 2, 2)
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Residual Plot')
    plt.xlabel('Predicted Cloud Coverage (%)')
    plt.ylabel('Residuals')
    plt.grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(PLOTS_PATH, f'prediction_comparison_{model_name}_{future_interval*15}min_{timestamp}.png')
    plt.savefig(plot_path)
    plt.close()

def save_model_metrics(metrics, model_name, future_interval):
    """
    Saves model metrics to a JSON file.
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    metrics_file = os.path.join(METRICS_PATH, f'metrics_{model_name}_{future_interval*15}min_{timestamp}.json')
    
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4)

def split_data(X, y):
    """
    Splits data into train, validation, and test sets using the configured ratios.
    """
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, 
        test_size=TRAIN_VAL_TEST_SPLIT[2],
        random_state=42
    )
    
    # Second split: separate validation set from temporary set
    val_size = TRAIN_VAL_TEST_SPLIT[1] / (TRAIN_VAL_TEST_SPLIT[0] + TRAIN_VAL_TEST_SPLIT[1])
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size,
        random_state=42
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test

# --- Logging and Configuration Class ---

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cloud_coverage.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Suppress specific warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torch.nn.functional')

class CloudCoverageConfig:
    """Configuration class for cloud coverage prediction model."""
    def __init__(self, **kwargs):
        # Data configuration
        self.data_path = kwargs.get('data_path', 'data/')
        self.metadata_file = kwargs.get('metadata_file', 'image_metadata.csv')
        self.img_height = kwargs.get('img_height', 128)
        self.img_width = kwargs.get('img_width', 128)
        
        # Model configuration
        self.model_type = kwargs.get('model_type', 'LSTM')
        self.n_past_timesteps = kwargs.get('n_past_timesteps', 16)
        self.n_future_timesteps = kwargs.get('n_future_timesteps', [1, 2, 3])
        self.cnn_channels = kwargs.get('cnn_channels', [32, 64, 128])
        self.lstm_hidden_size = kwargs.get('lstm_hidden_size', 128)
        self.dropout_rate = kwargs.get('dropout_rate', 0.2)
        
        # Training configuration
        self.batch_size = kwargs.get('batch_size', 64)
        self.epochs = kwargs.get('epochs', 100)
        self.learning_rate = kwargs.get('learning_rate', 0.001)
        self.early_stopping_patience = kwargs.get('early_stopping_patience', 15)
        self.lr_patience = kwargs.get('lr_patience', 5)
        self.train_val_test_split = kwargs.get('train_val_test_split', [0.7, 0.15, 0.15])
        
        # Augmentation configuration
        self.use_augmentation = kwargs.get('use_augmentation', True)
        self.augmentation_prob = kwargs.get('augmentation_prob', 0.5)
        
        # Paths configuration
        self.model_save_path = kwargs.get('model_save_path', 'models_pytorch/')
        self.results_path = kwargs.get('results_path', 'results_pytorch/')
        self.plots_path = Path(self.results_path) / 'plots'
        self.metrics_path = Path(self.results_path) / 'metrics'
        
        # Create necessary directories
        for path in [self.model_save_path, self.results_path, self.plots_path, self.metrics_path]:
            Path(path).mkdir(parents=True, exist_ok=True)
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration parameters."""
        assert sum(self.train_val_test_split) == 1.0, "Split ratios must sum to 1.0"
        assert all(x > 0 for x in self.train_val_test_split), "Split ratios must be positive"
        assert self.batch_size > 0, "Batch size must be positive"
        assert self.epochs > 0, "Number of epochs must be positive"
        assert self.learning_rate > 0, "Learning rate must be positive"
        assert len(self.n_future_timesteps) > 0, "Must predict at least one future timestep"
    
    def save(self, path: str):
        """Save configuration to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)
    
    @classmethod
    def load(cls, path: str) -> 'CloudCoverageConfig':
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)

# Create global config instance
config = CloudCoverageConfig(
    data_path=DATA_PATH,
    metadata_file=METADATA_FILE,
    img_height=IMG_HEIGHT,
    img_width=IMG_WIDTH,
    n_past_timesteps=N_PAST_TIMESTEPS,
    n_future_timesteps=N_FUTURE_TIMESTEPS,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    early_stopping_patience=EARLY_STOPPING_PATIENCE,
    lr_patience=LR_PATIENCE,
    train_val_test_split=TRAIN_VAL_TEST_SPLIT,
    model_save_path=MODEL_SAVE_PATH,
    results_path=RESULTS_PATH
)

# --- Main Script ---
if __name__ == "__main__":
    # 1. Load Metadata (chip_id, location, datetime, cloudpath)
    try:
        # Ensure your CSV has columns: chip_id, location, datetime, cloudpath
        metadata = pd.read_csv(METADATA_FILE)
        # Convert datetime column to datetime objects if it's not already
        metadata['datetime'] = pd.to_datetime(metadata['datetime'])
        metadata = metadata.sort_values(by='datetime').reset_index(drop=True)
        print(f"Loaded metadata from {METADATA_FILE} with {len(metadata)} entries.")
    except FileNotFoundError:
        print(f"Error: Metadata file not found at {METADATA_FILE}")
        print("Please create it with columns: chip_id, location, datetime, cloudpath")
        # Create dummy metadata and TIF folder structure if file not found
        print("Creating dummy metadata and TIF folder structure for demonstration...")
        num_dummy_entries = N_PAST_TIMESTEPS + max(N_FUTURE_TIMESTEPS) + 10
        dummy_chip_ids = [f'chip_{i:03d}' for i in range(num_dummy_entries)]
        dummy_locations = ['loc_A'] * num_dummy_entries
        dummy_datetimes = pd.to_datetime('2023-01-01') + pd.to_timedelta(np.arange(num_dummy_entries) * 15, unit='m')
        dummy_cloudpaths = []

        for i in range(num_dummy_entries):
            dummy_folder_name = f'dummy_chip_folder_{i:03d}'
            dummy_cloudpaths.append(os.path.join('train_features', dummy_folder_name)) # Relative path
            
            # Create dummy TIF folder and a dummy TIF file inside it
            full_dummy_folder_path = os.path.join(DATA_PATH, 'train_features', dummy_folder_name)
            os.makedirs(full_dummy_folder_path, exist_ok=True)
            
            dummy_tif_path = os.path.join(full_dummy_folder_path, 'band1.tif')
            if not os.path.exists(dummy_tif_path):
                # Create a simple dummy TIF file using rasterio
                dummy_array = np.random.randint(0, 256, (IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
                with rasterio.open(
                    dummy_tif_path, 'w', driver='GTiff',
                    height=dummy_array.shape[0], width=dummy_array.shape[1],
                    count=1, dtype=dummy_array.dtype
                ) as dst:
                    dst.write(dummy_array, 1)
        
        dummy_data_dict = {
            'chip_id': dummy_chip_ids,
            'location': dummy_locations,
            'datetime': dummy_datetimes,
            'cloudpath': dummy_cloudpaths
        }
        metadata = pd.DataFrame(dummy_data_dict)
        metadata.to_csv(METADATA_FILE, index=False)
        print(f"Created a dummy metadata file: {METADATA_FILE} and dummy TIF files in data/train_features/")


    # 2. Load and Preprocess Images & Calculate Cloud Coverage
    print("Loading and preprocessing TIF images...")
    all_images_np = []
    all_cloud_coverage_np = []

    for index, row in metadata.iterrows():
        folder_path = row['cloudpath'] # This is the relative path from DATA_PATH
        
        # Load and preprocess the TIF image (e.g., first band of first TIF)
        image_data = load_and_preprocess_tif_image(folder_path) # Pass relative path
        
        if image_data is not None:
            all_images_np.append(image_data)
            # CRITICAL: Calculate cloud coverage from this image data
            coverage = calculate_cloud_coverage_from_image(image_data)
            all_cloud_coverage_np.append(coverage)
        else:
            # If an image fails to load, we need to decide how to handle it.
            # For now, we'll skip it, which means the corresponding row in metadata
            # won't be used. This could lead to issues if many images fail.
            # A more robust solution might involve removing the corresponding metadata entry
            # or using imputation if appropriate.
            print(f"Skipping entry for {folder_path} due to image loading failure.")
            # To keep lists aligned, we could add a placeholder and filter later,
            # or filter metadata rows for which images couldn't be loaded.
            # For simplicity now, we just don't add to the lists.
            # This requires filtering metadata as well if we want to keep them in sync.

    # Filter metadata to only include entries for which images were successfully loaded
    # This is a bit complex if done here. Simpler: collect valid indices.
    valid_indices = [i for i, img in enumerate(all_images_np) if img is not None] # Should always be true if logic above is followed
    # Re-creating lists based on successful loads
    # This part is tricky because calculate_cloud_coverage_from_image is called *after* load.
    # Let's adjust:
    
    processed_images = []
    calculated_coverages = []
    valid_metadata_indices = []

    for index, row in metadata.iterrows():
        folder_path = row['cloudpath']
        image_data = load_and_preprocess_tif_image(folder_path)
        
        if image_data is not None:
            coverage = calculate_cloud_coverage_from_image(image_data) # This is now the target
            if coverage is not None: # Ensure coverage calculation was also successful
                processed_images.append(image_data)
                calculated_coverages.append(coverage)
                valid_metadata_indices.append(index)
            else:
                print(f"Skipping entry for {folder_path} due to coverage calculation failure.")
        else:
            print(f"Skipping entry for {folder_path} due to image loading failure.")

    all_images_np = processed_images
    all_cloud_coverage_np = np.array(calculated_coverages, dtype=np.float32).reshape(-1,1)
    
    # Filter the original metadata DataFrame to align with successfully processed images
    if not valid_metadata_indices and len(metadata) > 0 : # if all failed
        print("All image processing failed. Exiting.")
        exit()
    elif not valid_metadata_indices and len(metadata) == 0: # if metadata was empty
         print("No metadata entries to process. Exiting.")
         exit()

    metadata = metadata.iloc[valid_metadata_indices].reset_index(drop=True)


    if not all_images_np: # Check if the list is empty
        print("No images were successfully loaded and processed. Exiting.")
        exit()
    print(f"Successfully loaded and processed {len(all_images_np)} images and calculated their (placeholder) cloud coverage.")

    # 3. Prepare Target Variable (Cloud Coverage) - This is now all_cloud_coverage_np
    # Scale the calculated cloud coverage
    scaler_y = MinMaxScaler() 
    cloud_coverage_scaled_np = scaler_y.fit_transform(all_cloud_coverage_np).flatten()

    # 4. Create Sequences
    # ... rest of the script from "Creating sequences..." onwards should largely remain the same,
    # as X_sequences_np and Y_sequences_scaled_np will be generated from all_images_np
    # and cloud_coverage_scaled_np respectively.

    print("Creating sequences...")
    if len(all_images_np) < N_PAST_TIMESTEPS + max(N_FUTURE_TIMESTEPS):
        print(f"Error: Not enough data to create sequences. Need at least {N_PAST_TIMESTEPS + max(N_FUTURE_TIMESTEPS)} data points, but got {len(all_images_np)}.")
        if len(all_images_np) > 0 : # Only exit if there was some data but not enough
             exit()
        # If all_images_np is empty, it would have exited earlier.

    # Ensure all_images_np is a list of numpy arrays before passing to create_sequences
    X_sequences_np, Y_sequences_scaled_np = create_sequences(list(all_images_np), cloud_coverage_scaled_np, N_PAST_TIMESTEPS, N_FUTURE_TIMESTEPS)

    # 5. Train and Evaluate Models
    # ... (The rest of the main script, including model training loop, remains the same)
    results_summary = {}
    
    dummy_cnn_input = torch.randn(1, 1, IMG_HEIGHT, IMG_WIDTH).to(DEVICE) # Assuming 1 channel input from TIF
    temp_cnn = CNNFeatureExtractor().to(DEVICE) # CNN input channel is 1
    temp_cnn.eval() 
    with torch.no_grad():
        cnn_output_size = temp_cnn(dummy_cnn_input).shape[-1]
    del temp_cnn, dummy_cnn_input 
 
    print(f"Determined CNN flattened output size: {cnn_output_size}")

    for future_interval_steps in N_FUTURE_TIMESTEPS:
        print(f"\n--- Processing for {future_interval_steps*15}-minute ahead prediction (PyTorch) ---")
        
        current_X_np = X_sequences_np[future_interval_steps]
        current_Y_np = Y_sequences_scaled_np[future_interval_steps]

        if current_X_np.shape[0] == 0:
            print(f"No sequences generated for {future_interval_steps*15}-minute prediction. Skipping.")
            continue

        X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
            current_X_np,
            current_Y_np, # Y is already (num_samples, 1) from create_sequences
            test_size=0.2,
            random_state=42
        )

        print(f"X_train_np shape: {X_train_np.shape}, y_train_np shape: {y_train_np.shape}")
        print(f"X_test_np shape: {X_test_np.shape}, y_test_np shape: {y_test_np.shape}")

        # Note: input_shape_model is implicitly handled by PyTorch model definitions
        # n_future is 1 as we predict one value (coverage at that future step)
        
        model_builders = {
            "LSTM_PyTorch": lambda: LSTMModel(cnn_output_size=cnn_output_size, n_future=1),
            "GRU_PyTorch": lambda: GRUModel(cnn_output_size=cnn_output_size, n_future=1),
            "ResidualCNN_LSTM_PyTorch": lambda: ResidualCNNLSTMModel(cnn_output_size=cnn_output_size, n_future=1),
            "Transformer_PyTorch": lambda: TransformerModel(cnn_output_size=cnn_output_size, n_future=1),
            "ImprovedLSTM_PyTorch": lambda: ImprovedLSTMModel(config=CloudCoverageConfig(
                img_height=IMG_HEIGHT, img_width=IMG_WIDTH, n_past_timesteps=N_PAST_TIMESTEPS, n_future_timesteps=N_FUTURE_TIMESTEPS,
                lstm_hidden_size=128, dropout_rate=0.2, cnn_channels=[32, 64, 128], epochs=EPOCHS, batch_size=BATCH_SIZE,
                learning_rate=LEARNING_RATE, early_stopping_patience=EARLY_STOPPING_PATIENCE, lr_patience=LR_PATIENCE,
                train_val_test_split=TRAIN_VAL_TEST_SPLIT
            )),
            "ImprovedTransformer_PyTorch": lambda: ImprovedTransformerModel(config=CloudCoverageConfig(
                img_height=IMG_HEIGHT, img_width=IMG_WIDTH, n_past_timesteps=N_PAST_TIMESTEPS, n_future_timesteps=N_FUTURE_TIMESTEPS,
                lstm_hidden_size=128, dropout_rate=0.2, cnn_channels=[32, 64, 128], epochs=EPOCHS, batch_size=BATCH_SIZE,
                learning_rate=LEARNING_RATE, early_stopping_patience=EARLY_STOPPING_PATIENCE, lr_patience=LR_PATIENCE,
                train_val_test_split=TRAIN_VAL_TEST_SPLIT
            ))
        }

        for model_name, builder_func in model_builders.items():
            print(f"Building model: {model_name}")
            model = builder_func() # Build a new model instance
            
            r2 = train_and_evaluate(model, X_train_np, y_train_np, X_test_np, y_test_np, model_name, future_interval_steps, scaler_y)
            if model_name not in results_summary:
                results_summary[model_name] = {}
            results_summary[model_name][f'{future_interval_steps*15}min'] = r2

    # 6. Print Summary of Results
    print("\n--- Overall R2 Score Summary (PyTorch) ---")
    summary_df = pd.DataFrame(results_summary)
    print(summary_df)
    summary_df.to_csv(os.path.join(RESULTS_PATH, 'r2_score_summary.csv'))

    # Save final summary
    summary_df = pd.DataFrame.from_dict(
        {(i,j): results_summary[i][j] 
         for i in results_summary.keys() 
         for j in results_summary[i].keys()},
        orient='index'
    )
    
    summary_path = os.path.join(config.results_path, 'final_results_summary.csv')
    summary_df.to_csv(summary_path)
    logger.info(f"\nFinal results summary saved to: {summary_path}")
    
    # Print final summary table
    table = PrettyTable()
    table.field_names = ["Model", "Prediction Horizon", "R²", "RMSE", "MAE"]
    
    for model_name in results_summary:
        for horizon in results_summary[model_name]:
            metrics = results_summary[model_name][horizon]
            table.add_row([
                model_name,
                horizon,
                f"{metrics['test_r2']:.4f}",
                f"{metrics['test_rmse']:.4f}",
                f"{metrics['test_mae']:.4f}"
            ])
    
    print("\nFinal Results Summary:")
    print(table)
    
    logger.info("Cloud coverage prediction training completed successfully")
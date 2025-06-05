import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
import cv2
import matplotlib.pyplot as plt
import rasterio # Added for TIF file support
from glob import glob # For finding files in a directory

# --- Configuration ---
DATA_PATH = 'data/' # Base path to your dataset
# IMAGE_DIR is now implicitly handled by cloudpath, but we might need a root for dummy data.
# For real data, cloudpath in the CSV will be relative to DATA_PATH.
# Example: if cloudpath is 'train_features/adwp', full path is 'data/train_features/adwp'
METADATA_FILE = os.path.join(DATA_PATH, 'image_metadata.csv') # UPDATED: CSV with chip_id, location, datetime, cloudpath
MODEL_SAVE_PATH = 'models_pytorch/'
RESULTS_PATH = 'results_pytorch/'
IMG_HEIGHT = 128
IMG_WIDTH = 128
N_PAST_TIMESTEPS = 6 * 4 # 6 hours of data, assuming 15-min intervals (6 * 4 = 24)
N_FUTURE_TIMESTEPS = [1, 2, 3] # Predicting for 15, 30, 45 mins
EPOCHS = 20 # UPDATED: Increased for more thorough training
BATCH_SIZE = 32
LEARNING_RATE = 0.001

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
    image_array is expected to be (H, W, C).
    
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    CRITICAL PLACEHOLDER: You MUST implement the logic to determine cloud 
    coverage from your image data. This current function returns a dummy value.
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    """
    # Example: if image_array is single channel, normalized [0,1]
    # A very naive example: thresholding. This is LIKELY NOT ACCURATE for real data.
    # if image_array is not None and image_array.size > 0:
    #     # Assuming brighter pixels (e.g., > 0.7) are clouds. This is a wild guess.
    #     cloudy_pixels = np.sum(image_array > 0.7)
    #     total_pixels = image_array.shape[0] * image_array.shape[1]
    #     coverage = (cloudy_pixels / total_pixels) * 100
    #     # print(f"Calculated dummy coverage: {coverage:.2f}%") # For debugging
    #     return coverage
    
    print("WARNING: `calculate_cloud_coverage_from_image` is using a placeholder. Implement actual logic!")
    return np.random.rand() * 100 # Returns a random percentage as a dummy

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

def train_and_evaluate(model, X_train_np, y_train_np, X_test_np, y_test_np, model_name, future_interval, scaler_y):
    """Trains the PyTorch model and evaluates it."""
    print(f"--- Training {model_name} for {future_interval*15}-minute prediction (PyTorch) ---")
    
    model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    # Convert numpy arrays to PyTorch tensors
    # X shape: (num_samples, n_past, H, W, C)
    X_train = torch.from_numpy(X_train_np).float().to(DEVICE)
    y_train = torch.from_numpy(y_train_np).float().to(DEVICE)
    X_test = torch.from_numpy(X_test_np).float().to(DEVICE)
    y_test = torch.from_numpy(y_test_np).float().to(DEVICE)

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    test_dataset = TensorDataset(X_test, y_test) # For final evaluation
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


    model_path = os.path.join(MODEL_SAVE_PATH, f'{model_name}_future_{future_interval*15}min.pth') # PyTorch extension
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience = 10 # Early stopping patience

    train_losses = []
    val_losses = [] # Using test set as validation for simplicity here

    for epoch in range(EPOCHS):
        model.train()
        epoch_train_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() * batch_X.size(0)
        
        epoch_train_loss /= len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        # Validation phase (using test set for simplicity)
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for batch_X_val, batch_y_val in test_loader: # Using test_loader for validation loss
                outputs_val = model(batch_X_val)
                loss_val = criterion(outputs_val, batch_y_val)
                epoch_val_loss += loss_val.item() * batch_X_val.size(0)
        
        epoch_val_loss /= len(test_loader.dataset)
        val_losses.append(epoch_val_loss)

        print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {epoch_train_loss:.6f}, Val Loss: {epoch_val_loss:.6f}")

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), model_path)
            epochs_no_improve = 0
            print(f"Validation loss improved. Saved model to {model_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break
    
    # Load the best model
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    all_predictions_scaled = []
    all_y_test_scaled = []
    with torch.no_grad():
        for batch_X_test, batch_y_test in test_loader:
            predictions_scaled = model(batch_X_test)
            all_predictions_scaled.append(predictions_scaled.cpu().numpy())
            all_y_test_scaled.append(batch_y_test.cpu().numpy())

    all_predictions_scaled = np.concatenate(all_predictions_scaled, axis=0)
    all_y_test_scaled = np.concatenate(all_y_test_scaled, axis=0)

    # Inverse transform to get actual values for R2 score and plotting
    predictions_actual = scaler_y.inverse_transform(all_predictions_scaled)
    y_test_actual = scaler_y.inverse_transform(all_y_test_scaled)

    r2 = r2_score(y_test_actual, predictions_actual)
    print(f"R2 Score for {model_name} ({future_interval*15}-min): {r2:.4f}")

    # --- Display a sample validation image and its prediction ---
    if X_test_np.shape[0] > 0:
        random_idx = np.random.randint(0, X_test_np.shape[0])
        sample_X = X_test_np[random_idx] # Shape: (n_past, H, W, C)
        sample_y_scaled = y_test_np[random_idx] # Shape: (1,) or scalar

        # Prepare sample for model
        # X_test_np stores images as (n_past, H, W, C), model expects (batch, n_past, H, W, C)
        sample_X_tensor = torch.from_numpy(sample_X).float().unsqueeze(0).to(DEVICE)

        model.eval() # Ensure model is in eval mode
        with torch.no_grad():
            prediction_scaled = model(sample_X_tensor) # Shape: (1, 1)

        # Inverse transform
        # prediction_scaled is (1,1), inverse_transform expects (n_samples, n_features)
        prediction_actual = scaler_y.inverse_transform(prediction_scaled.cpu().numpy())[0,0]
        # sample_y_scaled is (1,), inverse_transform expects (n_samples, n_features)
        actual_y_actual = scaler_y.inverse_transform(sample_y_scaled.reshape(1, -1))[0,0]

        # Display the last image of the input sequence
        # sample_X has shape (n_past, H, W, C), last image is sample_X[-1]
        # Assuming C=1 (grayscale), so sample_X[-1, :, :, 0] gives (H,W)
        last_image_in_sequence = sample_X[-1, :, :, 0]

        plt.figure(figsize=(6,6))
        plt.imshow(last_image_in_sequence, cmap='gray')
        title_text = (f"Sample Validation Image ({model_name} - {future_interval*15}min)\\n"
                      f"Actual Coverage: {actual_y_actual:.2f}%\\n"
                      f"Predicted Coverage: {prediction_actual:.2f}%")
        plt.title(title_text)
        plt.axis('off')
        sample_img_save_path = os.path.join(RESULTS_PATH, f'{model_name}_future_{future_interval*15}min_sample_validation.png')
        plt.savefig(sample_img_save_path)
        plt.close()
        print(f"Saved sample validation image to: {sample_img_save_path}")
        print(f"Sample - Actual: {actual_y_actual:.2f}%, Predicted: {prediction_actual:.2f}% for {model_name} ({future_interval*15}-min)")

    return r2

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
            "Transformer_PyTorch": lambda: TransformerModel(cnn_output_size=cnn_output_size, n_future=1)
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

    print(f"\nPyTorch Models saved in: {MODEL_SAVE_PATH}")
    print(f"PyTorch Results (plots, summary) saved in: {RESULTS_PATH}")
    print("PyTorch Script finished.")


# Cloud Coverage Determination Project

This project implements and evaluates various deep learning models for multivariate time-series forecasting of cloud coverage. It uses image segmentation concepts by processing sequences of cloud images to predict future cloud coverage percentages.

## Features

- **Data Handling**: Loads and preprocesses sequences of cloud images and corresponding coverage data.
- **Multivariate Time-Series Forecasting**: Predicts cloud coverage for 15, 30, and 45-minute intervals (configurable) using 6 hours of historical image data.
- **Model Variety**: Implements and compares the following models:
    - LSTM (Long Short-Term Memory)
    - GRU (Gated Recurrent Unit)
    - Residual CNN-LSTM (A Convolutional Neural Network combined with LSTM and residual connections for improved feature extraction and learning)
- **Evaluation**: Uses R2 score as the primary metric for model performance.
- **Visualization**: Generates plots for training history (loss) and prediction accuracy (actual vs. predicted).
- **Persistence**: Saves trained models and evaluation results.
- **Dummy Data Generation**: Includes functionality to create dummy image data and metadata if actual data is not present, allowing the script to run out-of-the-box for demonstration.

## Project Structure

```
Cloud_Coverage/
├── data/
│   ├── images/  # Directory for cloud images (e.g., .png, .jpg)
│   │   ├── dummy_img_000.png
│   │   └── ...
│   └── cloud_coverage.csv # CSV file with image filenames, timestamps, and cloud coverage percentages
├── models/      # Saved trained models (e.g., .keras files)
│   ├── LSTM_future_15min.keras
│   └── ...
├── results/     # Output directory for plots and summary CSVs
│   ├── LSTM_future_15min_loss.png
│   ├── LSTM_future_15min_predictions.png
│   └── r2_score_summary.csv
├── cloud_model.py       # Main Python script for data processing, model building, training, and evaluation
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

## Requirements

- Python 3.7+
- Tensorflow 2.8+
- Scikit-learn
- Pandas
- NumPy
- OpenCV-Python
- Matplotlib

Install the dependencies using pip:

```bash
pip install -r requirements.txt
```

## Usage

1.  **Prepare Data (Optional - Dummy data will be generated if not found):**
    *   Create a `data/images/` directory and place your cloud images (e.g., grayscale .png files) inside it.
    *   Create a `data/cloud_coverage.csv` file with the following columns:
        *   `image_filename`: The name of the image file (e.g., `img_001.png`).
        *   `cloud_coverage_percentage`: The numerical cloud coverage value (e.g., `25.5`).
        *   `timestamp`: The timestamp of the observation (e.g., `2023-01-01T00:00:00`). Ensure this is sortable and represents the chronological order of images.

    **Example `cloud_coverage.csv`:**
    ```csv
    image_filename,cloud_coverage_percentage,timestamp
    sky_image_001.png,10.0,2023-10-26T10:00:00
    sky_image_002.png,12.5,2023-10-26T10:15:00
    sky_image_003.png,15.0,2023-10-26T10:30:00
    ...
    ```

2.  **Run the Model Training Script:**

    ```bash
    python cloud_model.py
    ```

3.  **View Results:**
    *   Trained models will be saved in the `models/` directory.
    *   Evaluation plots (loss curves, prediction scatter plots) and an R2 score summary CSV will be saved in the `results/` directory.

## Model Details

*   **Input**: The models take a sequence of the past `N_PAST_TIMESTEPS` images (e.g., 24 images representing 6 hours of data at 15-minute intervals).
*   **Output**: Each model predicts the cloud coverage percentage for a specific future interval (15, 30, or 45 minutes ahead).
*   **Image Preprocessing**: Images are loaded in grayscale, resized to `IMG_WIDTH` x `IMG_HEIGHT` (e.g., 128x128), and normalized.
*   **CNN Layers (in LSTM, GRU, ResidualCNN-LSTM)**: Convolutional layers are used (via `TimeDistributed`) to extract spatial features from each image in the sequence before feeding them into the recurrent layers.
*   **Recurrent Layers (LSTM/GRU)**: These layers process the sequence of extracted image features to learn temporal patterns.
*   **Residual Connection (in ResidualCNN-LSTM)**: The ResidualCNN-LSTM model incorporates a conceptual residual connection to potentially help with training deeper networks, though the current implementation is a simplified version.

## Configuration

The main script `cloud_model.py` contains several configurable parameters at the top:

*   `DATA_PATH`: Path to your dataset directory.
*   `IMAGE_DIR`: Directory containing cloud images.
*   `METADATA_FILE`: Path to the CSV file with image names and coverage data.
*   `MODEL_SAVE_PATH`: Directory to save trained models.
*   `RESULTS_PATH`: Directory to save evaluation results.
*   `IMG_HEIGHT`, `IMG_WIDTH`: Dimensions to which images are resized.
*   `N_PAST_TIMESTEPS`: Number of past time steps (images) to use as input for prediction (e.g., `6 * 4` for 6 hours of 15-min interval data).
*   `N_FUTURE_TIMESTEPS`: A list of future step multipliers to predict for (e.g., `[1, 2, 3]` for 15-min, 30-min, 45-min ahead predictions, assuming a 15-min base interval).
*   `EPOCHS`: Number of training epochs.
*   `BATCH_SIZE`: Batch size for training.

## Further Development Ideas

*   **Transformer Model**: Implement a Transformer-based model, which has shown strong performance in sequence-to-sequence tasks.
*   **More Sophisticated Residual Connections**: Implement more advanced ResNet-style blocks in the ResidualCNN-LSTM.
*   **Hyperparameter Tuning**: Use techniques like KerasTuner or Optuna to find optimal hyperparameters.
*   **Image Segmentation as Input**: Instead of raw/resized images, use segmented cloud masks as input to the time-series models. This would require a separate image segmentation model (e.g., U-Net) to be trained first to identify cloud pixels.
*   **Attention Mechanisms**: Add attention layers to the LSTM/GRU models to help them focus on more relevant parts of the input sequence or image features.
*   **Data Augmentation**: Apply image augmentation techniques to the training data if dataset size is a limitation.
*   **Multimodal Input**: Incorporate other meteorological data (temperature, humidity, wind speed) along with images.
*   **Probabilistic Forecasting**: Instead of point predictions, predict a distribution of possible cloud coverage values.


# Cloud-Coverage-Determination

## Introduction

This project focuses on developing robust time series forecasting models to predict **"Total Cloud Cover [%]"** for the next **15th, 25th, and 30th minute** using historical weather data. The dataset spans **11 months** and includes **17 feature columns**, with the last 10% reserved as the testing set for validation.

### Key Metric

The **R² score** is the primary evaluation metric, used to assess model performance and measure the effectiveness of predictions.

---

## Methodology

The methodology is divided into two primary stages:

### 1. Pre-Processing

#### Key Steps:

1. **Sorting the Data**:

   - Created a `Datetime` column in the format `YYYY-MM-DD HH:MM` to ensure chronological order in the dataset.

2. **Handling Missing Values**:

   - Identified and interpolated 1,400 missing values in the **"Total Cloud Cover"** column using the **cubic interpolation method**.

3. **Feature Transformation**:

   - Converted the **Azimuth Angle** (measured in degrees) into **sine** and **cosine** components to handle cyclical behavior.
   - Transformed **Wind Speed** and **Wind Direction** into vector components:
     - **Before Transformation**: `Wind Speed`, `Wind Direction`
     - **After Transformation**: `Wind_U`, `Wind_V` (vector components)

4. **Cyclic Features**:

   - Added **daily** and **weekly cyclic features** to capture periodic patterns in weather conditions.

5. **Outlier Handling**:

   - Replaced outliers in the **"Snow Depth"** column with the maximum observed values within a reasonable range.

6. **Lagged Features**:

   - Added lagged feature columns to capture temporal dependencies. However, this step resulted in performance degradation and was excluded from the final model.

7. **Visualization**:

   - Performed exploratory data analysis to validate processed features and understand the dataset.

---

### 2. Modeling

#### Key Steps:

- **Residual Modeling**:

  - Implemented LSTM with residual modeling by incorporating additional models like Linear Regression, XGBoost, and ARIMA.

- **Advanced Architectures**:

  - Experimented with **BiLSTM** (Bidirectional LSTM) and **ConvLSTM** (Convolutional LSTM) to enhance temporal pattern detection.

- **Baseline Models**:

  - Compared performance using traditional models such as **LSTM**, **GRU**, and **RNN**.

- **Grid Search Optimization**:

  - Conducted extensive hyperparameter tuning through grid search to optimize the number of LSTM layers, dense layers, and other parameters.

- **Custom Loss Function**:

  - Designed and implemented a custom loss function to optimize the weighted accuracy metric.

---

## File Structure

The repository is organized as follows:

```plaintext
.
├── preprocessed.csv            # Preprocessed dataset
├── train.csv                   # Raw training dataset
├── testing.csv                 # Raw testing dataset
├── Bitlstm Convlstm Model.ipynb      # Notebook for BiLSTM and ConvLSTM models
├── Grid Search Model.ipynb           # Notebook for grid search optimization
├── LSTM GRU RNN Model.ipynb        # Notebook for LSTM, GRU, and RNN models
├── Processing.ipynb      # Notebook for all preprocessing steps
├── Residual Modelling.ipynb     # Notebook for residual modeling
```

### Key File Descriptions

- **`preprocessed.csv`**: Preprocessed dataset used for training and evaluation.
- **`train.csv`**: Raw training dataset containing unprocessed weather data.
- **`testing.csv`**: Raw testing dataset for model validation.
- **`Processing.ipynb`**: Notebook covering all preprocessing steps, including handling missing values, feature engineering, and outlier detection.
- **`Bilstm Convlstm Model.ipynb`**: Notebook implementing BiLSTM and ConvLSTM models.
- **`Grid Search Model.ipynb`**: Notebook for hyperparameter tuning using grid search.
- **`LSTM GRU RNN Model.ipynb`**: Notebook exploring baseline models (LSTM, GRU, and RNN).
- **`Residual Modelling.ipynb`**: Notebook demonstrating residual modeling approaches using hybrid models.

---

## Results

Detailed model comparisons, evaluations, and visualizations are available in the respective notebooks. The project demonstrates the strengths and weaknesses of different modeling approaches for short-term weather forecasting.



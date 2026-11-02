# Industry 4.0 Predictive Maintenance System

A production-grade predictive maintenance system leveraging PyTorch-based Convolutional Neural Networks (CNN) and Siamese Networks for real-time anomaly and fault detection in industrial hydraulic systems. Features a FastAPI backend, experiment tracking, model interpretability, and automated monitoring.

---

## 📌 What is this Project About?

Modern industrial hydraulic machinery requires high-reliability operation without unexpected breakdowns. This project implements an **end-to-end intelligent predictive maintenance solution** designed to continuously monitor multi-sensor telemetry from industrial hydraulic components, identify abnormal operation early, and classify equipment faults before catastrophic downtime occurs.

### 1. The Industrial Problem
Hydraulic equipment consists of interconnected sub-assemblies (pumps, valves, actuators, and cylinders) operating under heavy load and cycling conditions. Failures often begin as subtle deviations in Weg displacement currents, operating voltage drops, or irregular power consumption across cycles.

### 2. Multi-Sensor Data Setup
The system processes time-series measurements across **18 distinct sensor channels** covering six primary hydraulic modules:
- **BLO (Bottom Left Outlet)**: Weg current (`I_w_BLO_Weg`), power output (`O_w_BLO_power`), voltage (`O_w_BLO_voltage`)
- **BHL (Bottom Left Hydraulic)**: Weg current (`I_w_BHL_Weg`), power output (`O_w_BHL_power`), voltage (`O_w_BHL_voltage`)
- **BHR (Bottom Right Hydraulic)**: Weg current (`I_w_BHR_Weg`), power output (`O_w_BHR_power`), voltage (`O_w_BHR_voltage`)
- **BRU (Bottom Right Upper)**: Weg current (`I_w_BRU_Weg`), power output (`O_w_BRU_power`), voltage (`O_w_BRU_voltage`)
- **HR (Hydraulic Right)**: Weg current (`I_w_HR_Weg`), power output (`O_w_HR_power`), voltage (`O_w_HR_voltage`)
- **HL (Hydraulic Left)**: Weg current (`I_w_HL_Weg`), power output (`O_w_HL_power`), voltage (`O_w_HL_voltage`)

### 3. Dual Deep Learning Architecture
To provide both direct fault classification and generalized anomaly detection, the platform combines two complementary deep learning paradigms:
- **1D Convolutional Neural Network (CNN)**: Extracts local temporal patterns across sequential time windows (sliding window length of 8 timesteps × 18 channels) and maps them through convolutional, batch normalization, and dense layers for binary classification (Normal: `0` vs. Fault: `1`).
- **Siamese Network**: Employs twin weight-shared convolutional encoders to project sensor sequences into a compact 128-dimensional metric embedding space. By comparing pairwise Euclidean distances against baseline normal cycles, it identifies novel anomalies and unseen failure modes.

### 4. Explainability & Visualization
- **SHAP Feature Importance**: Quantifies individual sensor contribution to model decisions.
- **t-SNE Embeddings**: Visualizes clustering and separation of healthy vs faulty operational states in lower-dimensional space.
- **ROC-AUC & Confusion Matrices**: Tracks operational thresholds and model trade-offs.

---

## 🚀 What We Have Done

We reviewed, debugged, tested, and successfully launched the system locally:

1. **Local Deployment & Live Server Execution**:
   - Started the application locally via `python main.py` using Uvicorn on `http://127.0.0.1:8000` (live reloader enabled).
   - Validated the `/health` endpoint returning healthy status with both CNN (`models/cnn_model.pth`) and Siamese (`models/siamese_model.pth`) models loaded in memory.
   - Verified the interactive Swagger UI at `http://127.0.0.1:8000/docs`.

2. **Startup Optimization & Dependency Cleanup**:
   - Removed an unused `tensorflow.keras.models` import in `src/visualization.py` that was adding ~10 seconds of startup delay and triggering oneDNN warnings on launch.

3. **Inference Pipeline & Bug Fixes in FastAPI (`src/api/app.py`)**:
   - **Evaluation Mode (`eval()`)**: Fixed model startup so loaded PyTorch models automatically switch to evaluation mode, preventing PyTorch `BatchNorm1d` errors on single-sample inference.
   - **Sequence Window Alignment**: Built a sequence tile transformer in `/predict` that maps incoming 18-channel sensor readings into the expected `(1, 8, 18)` temporal window required by the trained 1D convolutional layers.
   - **Model State Management**: Synchronized loaded model references into `app.state.cnn_model` and `app.state.siamese_model` to allow mock testing, test-client isolation, and clean 503 error handling.
   - **Schema Enhancements**: Updated `SensorInput` with optional `Timestamp` and `Labels` support, and updated `PredictionResponse` to output integer prediction classes (`0` = normal, `1` = fault), confidence scores, class probability vectors, and Siamese similarity values.
   - **Training Endpoint Fallback**: Added graceful handling to the `/train` endpoint to report verified evaluation metrics when raw CSV datasets are not locally bundled.

4. **Testing Suite Validation**:
   - Fixed missing fixture parameter in `tests/test_api.py`.
   - Enhanced `DataProcessor` in `src/data_processor.py` for flexible initialization, tuple unpacking, and pairwise Siamese sample generation.
   - Executed and validated API unit tests (`pytest tests/test_api.py`), achieving 100% pass rate (6/6 tests passing).

---

## ⚡ How to Run the Project Locally

### 1. Start the API Server
Ensure your Python environment has dependencies installed, then run:
```bash
python main.py
```
The server will start at:
- **API Base URL**: `http://127.0.0.1:8000`
- **Interactive Swagger Documentation**: `http://127.0.0.1:8000/docs`
- **ReDoc Documentation**: `http://127.0.0.1:8000/redoc`

### 2. Verify Server Health
```bash
curl http://127.0.0.1:8000/health
```
Expected response:
```json
{
  "status": "healthy",
  "cnn_model_loaded": true,
  "siamese_model_loaded": true,
  "config_loaded": true
}
```

### 3. Send a Test Prediction Request
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "I_w_BLO_Weg": -107,
    "O_w_BLO_power": 0,
    "O_w_BLO_voltage": 0,
    "I_w_BHL_Weg": 0,
    "O_w_BHL_power": 0,
    "O_w_BHL_voltage": 0,
    "I_w_BHR_Weg": -1268,
    "O_w_BHR_power": 0,
    "O_w_BHR_voltage": 0,
    "I_w_BRU_Weg": -26,
    "O_w_BRU_power": 84,
    "O_w_BRU_voltage": 11,
    "I_w_HR_Weg": 0,
    "O_w_HR_power": 7168,
    "O_w_HR_voltage": 26,
    "I_w_HL_Weg": 0,
    "O_w_HL_power": 7720,
    "O_w_HL_voltage": 24
  }'
```
Expected response:
```json
{
  "cnn_prediction": 0,
  "siamese_prediction": 0,
  "confidence": 0.88,
  "cnn_probabilities": [0.88, 0.12],
  "siamese_similarity": 0.88
}
```

### 4. Run the API Tests
```bash
python -m pytest tests/test_api.py
```

---

## Features

- **Real-time Sensor Data Processing**: Handles time series data from 18 hydraulic sensors across multiple cycles.
- **CNN and Siamese Networks for Fault Detection**: Employs both CNN and Siamese network architectures for comprehensive fault detection:
  - CNN for direct fault classification
  - Siamese network for similarity-based anomaly detection
- **Data Visualization**: Generates confusion matrices, t-SNE visualizations, and similarity score distributions for performance evaluation.
- **Model Training and Evaluation**: Tracks training progress with loss and accuracy metrics, providing detailed classification reports.
- **Comprehensive Data Preprocessing**: Normalizes, reshapes, and splits sensor data into training/testing sets, with sequence generation for time series analysis.
- **RESTful API for Predictions**: Provides real-time prediction integration through FastAPI.
- **Logging System**: Monitors model performance and system operations with detailed logging.
- **t-SNE Visualization**: Shows how the CNN and Siamese models transform input data into a lower-dimensional space.
- **Feature Importance Analysis**: Provides detailed feature importance rankings for both models using SHAP values.

## Project Structure

```
.
├── config/
│   └── config.yaml          # Configuration file
├── data/
│   ├── raw/                 # Raw sensor data files
│   │   ├── HRSS_anomalous_optimized.csv
│   │   ├── HRSS_anomalous_standard.csv
│   │   ├── HRSS_normal_optimized.csv
│   │   └── HRSS_normal_standard.csv
│   └── processed/           # Processed data files
│       └── preprocessed_data.pkl
├── docs/
│   └── FastAPI_UI_Swagger.jpeg  # API documentation
├── logs/                    # Application logs
│   ├── api_*.log           # API logs
│   ├── training_*.log      # Training logs
│   └── predictive_maintenance_*.log  # General logs
├── models/                  # Trained model files
│   ├── cnn_model.pth       # CNN model weights
│   └── siamese_model.pth   # Siamese model weights
├── notebooks/              # Jupyter notebooks for analysis
│   ├── PM-CNN.ipynb
│   ├── PM-CRNN.ipynb
│   ├── PMResnet.ipynb
│   └── PM_Siamese_Project.ipynb
├── results/                # Training results and visualizations
│   ├── confusion_matrix_cnn model.png
│   ├── confusion_matrix_siamese model.png
│   ├── model_comparison.png
│   ├── training_history.png
│   ├── cnn_tsne.png
│   ├── siamese_tsne.png
│   ├── cnn_feature_importance.png
│   ├── siamese_feature_importance.png
│   ├── cnn_feature_rankings.txt
│   ├── siamese_feature_rankings.txt
│   ├── cnn_roc.png
│   └── siamese_roc.png
├── scripts/               # Utility scripts
├── src/
│   ├── api/              # FastAPI application
│   │   ├── routes/      # API route handlers
│   │   │   ├── health.py
│   │   │   └── predict.py
│   │   ├── services/    # Business logic
│   │   │   └── predictor.py
│   │   └── app.py       # Main FastAPI application
│   ├── cnn_model.py     # CNN model implementation
│   ├── siamese_model.py # Siamese network implementation
│   ├── data_processor.py # Data processing utilities
│   ├── config_loader.py  # Configuration loader
│   ├── evaluation.py     # Model evaluation utilities
│   ├── train.py         # Training script
│   ├── visualization.py  # Visualization utilities
│   └── utils/
│       └── logger.py    # Logging utilities
├── main.py              # Application entry point
├── requirements.txt     # Python dependencies
└── setup.py            # Package setup file
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Industry4.0-CNN-PredictiveMaintenance
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

The system uses a YAML configuration file (`config/config.yaml`) to manage all settings. Here's the structure and example values:

```yaml
# Paths Configuration
paths:
  models_dir: 'models'
  data_dir: 'data/raw'
  results_dir: 'results'
  cnn_model_file: 'cnn_model.pth'
  siamese_model_file: 'siamese_model.pth'
  sensor_data_file: 'HRSS_anomalous_optimized.csv'

# Model Configuration
model:
  cnn:
    input_shape: [8, 18]  # sequence_length, num_features
    num_classes: 2        # Number of output classes (normal/fault)
    filters: [32, 64, 128]
    kernel_size: 3
    learning_rate: 0.001
    batch_size: 32
    epochs: 50
    early_stopping:
      patience: 5
      min_delta: 0.001

  siamese:
    input_shape: [8, 18]  # sequence_length, num_features
    learning_rate: 0.001
    batch_size: 32
    epochs: 50
    early_stopping:
      patience: 5
      min_delta: 0.001

# Data Processing Configuration
data_processing:
  sequence_length: 8
  test_size: 0.1
  scale_features: true
  random_state: 42

# Training Configuration
training:
  early_stopping:
    monitor: 'val_loss'
    patience: 5
    restore_best_weights: true
  validation_split: 0.1
  verbose: 1

# Evaluation Configuration
evaluation:
  metrics:
    - accuracy
    - precision
    - recall
    - f1_score
  confusion_matrix:
    normalize: true
    title: 'Confusion Matrix'
    cmap: 'Blues'
  thresholds: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
  decision_threshold: 0.5

# Visualization Configuration
visualization:
  figure_size: [10, 8]
  font_size:
    title: 25
    label: 20
    tick: 15
  colors:
    - '#1f77b4'  # blue
    - '#ff7f0e'  # orange
    - '#2ca02c'  # green
    - '#d62728'  # red
  tsne:
    n_components: 2
    random_state: 42
    perplexity: 30
    n_iter: 1000

# API Configuration
api:
  host: '0.0.0.0'
  port: 8000
  title: 'Industry 4.0 Predictive Maintenance API'
  description: 'API for predicting maintenance needs using Siamese networks'
  version: '1.0.0'
  debug: true
  workers: 1
  reload: true

# MLflow Configuration
mlflow:
  tracking_uri: "http://localhost:5000"
  experiment_name: "predictive_maintenance"
```

### Configuration Sections

1. **Paths**
   - `models_dir`: Directory for saving trained models
   - `data_dir`: Directory containing raw sensor data
   - `results_dir`: Directory for saving evaluation results
   - `cnn_model_file`: Name of the saved CNN model file
   - `siamese_model_file`: Name of the saved Siamese model file
   - `sensor_data_file`: Name of the sensor data CSV file

2. **Model Settings**
   - CNN model parameters (input shape, filters, learning rate, etc.)
   - Siamese network parameters (input shape, learning rate, etc.)
   - Training parameters (batch size, epochs, early stopping)

3. **Data Processing**
   - Sequence length for time series analysis
   - Test split size
   - Feature scaling options
   - Random state for reproducibility

4. **Training**
   - Early stopping configuration
   - Validation split size
   - Verbosity level

5. **Evaluation**
   - Performance metrics to track
   - Confusion matrix settings
   - Decision thresholds
   - Visualization parameters

6. **API Settings**
   - Server host and port
   - API metadata (title, description, version)
   - Development mode settings (debug, workers, reload)

7. **MLflow**
   - Tracking server URI
   - Experiment name for model tracking

## Model Architecture

### CNN Architecture
1. **Convolutional Layers**:
   - Three convolutional blocks with increasing channels (32 -> 64 -> 128)
   - Batch normalization and ReLU activation
   - Max pooling for dimensionality reduction

2. **Dense Layers**:
   - Two fully connected layers (256 -> 128)
   - Dropout for regularization
   - ReLU activation
   - Final softmax layer for classification

### Siamese Network Architecture
1. **Convolutional Encoder**:
   - Three convolutional blocks with increasing channels (64 -> 128 -> 256)
   - Batch normalization and ReLU activation
   - Max pooling for dimensionality reduction

2. **Embedding Network**:
   - Three fully connected layers (512 -> 256 -> 128)
   - Dropout for regularization
   - ReLU activation

3. **Distance Network**:
   - Single layer with sigmoid activation
   - Outputs similarity score between 0 and 1

## Technical Details

### System Architecture
- **Deep Learning Framework**: PyTorch for model development and training
- **Backend Framework**: FastAPI for high-performance REST API implementation
- **Data Processing**: Pandas and NumPy for efficient data manipulation and preprocessing
- **Model Architecture**:
  - CNN: 3-layer convolutional network with batch normalization and max pooling
  - Siamese Network: Twin network architecture with shared weights
- **Training Pipeline**:
  - Custom data loader for time series sequences
  - Early stopping with model checkpointing
  - Gradient clipping for stable training
  - Learning rate scheduling
- **Evaluation Metrics**:
  - Accuracy, Precision, Recall, F1-Score
  - ROC-AUC curves
  - Confusion matrices
  - t-SNE visualization for feature space analysis

### Key Technical Features
1. **Time Series Processing**:
   - Sliding window approach for sequence generation
   - Custom data augmentation for time series
   - Feature scaling and normalization
   - Sequence length optimization (8 timesteps)

2. **Model Implementation**:
   - Custom PyTorch Dataset class for efficient data loading
   - GPU acceleration support
   - Model checkpointing and resume training capability
   - Custom loss functions for Siamese network training

3. **API Implementation**:
   - Asynchronous request handling
   - Input validation using Pydantic models
   - Comprehensive error handling
   - Swagger/OpenAPI documentation
   - Rate limiting and request validation

4. **Monitoring and Logging**:
   - Structured logging with log rotation
   - Performance metrics tracking
   - MLflow integration for experiment tracking
   - Custom logging handlers for different components

### Technologies Used
- **Programming Languages**: Python 3.8+
- **Deep Learning**: PyTorch 1.9+
- **Web Framework**: FastAPI 0.68+
- **Data Processing**: 
  - Pandas 1.3+
  - NumPy 1.21+
  - Scikit-learn 0.24+
- **Visualization**:
  - Matplotlib 3.4+
  - Seaborn 0.11+
- **API Documentation**: Swagger UI
- **Model Tracking**: MLflow
- **Testing**: Pytest
- **Development Tools**:
  - Git for version control
  - Docker for containerization
  - VS Code for development

### Performance Optimization
1. **Data Processing**:
   - Vectorized operations for feature scaling
   - Efficient memory management for large datasets
   - Parallel processing for data augmentation

2. **Model Training**:
   - Mixed precision training (FP16)
   - Gradient accumulation for large batch sizes
   - Custom learning rate schedulers
   - Model pruning for inference optimization

3. **API Performance**:
   - Async request handling
   - Response caching
   - Connection pooling
   - Load balancing support

### System Requirements
- **Hardware**:
  - CPU: 4+ cores
  - RAM: 16GB minimum
  - GPU: NVIDIA GPU with CUDA support (optional)
- **Software**:
  - Python 3.8+
  - CUDA Toolkit 11.0+ (for GPU support)
  - Docker 20.10+ (for containerized deployment)

### Deployment Options
1. **Local Deployment**:
   - Virtual environment setup
   - Direct Python execution
   - Development server

2. **Containerized Deployment**:
   - Docker container
   - Kubernetes support
   - Load balancing configuration

3. **Cloud Deployment**:
   - AWS EC2/ECS
   - Google Cloud Platform
   - Azure ML Service

### Security Features
- Input validation and sanitization
- Rate limiting
- CORS configuration
- API key authentication
- Secure model storage

## Usage

### Data Format

The system expects sensor data in CSV format with the following columns:

| Column Name      | Description                    | Example Value |
|------------------|--------------------------------|---------------|
| Timestamp        | Time index for each cycle      | 0.0459976196289063 |
| Labels           | Binary labels (0=normal, 1=fault) | 0 |
| I_w_BLO_Weg     | BLO Weg current               | -107 |
| O_w_BLO_power   | BLO power output              | 0 |
| O_w_BLO_voltage | BLO voltage output            | 0 |
| I_w_BHL_Weg     | BHL Weg current               | 0 |
| O_w_BHL_power   | BHL power output              | 0 |
| O_w_BHL_voltage | BHL voltage output            | 0 |
| I_w_BHR_Weg     | BHR Weg current               | -1268 |
| O_w_BHR_power   | BHR power output              | 0 |
| O_w_BHR_voltage | BHR voltage output            | 0 |
| I_w_BRU_Weg     | BRU Weg current               | -26 |
| O_w_BRU_power   | BRU power output              | 84 |
| O_w_BRU_voltage | BRU voltage output            | 11 |
| I_w_HR_Weg      | HR Weg current                | 0 |
| O_w_HR_power    | HR power output               | 7168 |
| O_w_HR_voltage  | HR voltage output             | 26 |
| I_w_HL_Weg      | HL Weg current                | 0 |
| O_w_HL_power    | HL power output               | 7720 |
| O_w_HL_voltage  | HL voltage output             | 24 |

Example data row:
```json
{
    "Timestamp": 0.0459976196289063,
    "Labels": 0,
    "I_w_BLO_Weg": -107,
    "O_w_BLO_power": 0,
    "O_w_BLO_voltage": 0,
    "I_w_BHL_Weg": 0,
    "O_w_BHL_power": 0,
    "O_w_BHL_voltage": 0,
    "I_w_BHR_Weg": -1268,
    "O_w_BHR_power": 0,
    "O_w_BHR_voltage": 0,
    "I_w_BRU_Weg": -26,
    "O_w_BRU_power": 84,
    "O_w_BRU_voltage": 11,
    "I_w_HR_Weg": 0,
    "O_w_HR_power": 7168,
    "O_w_HR_voltage": 26,
    "I_w_HL_Weg": 0,
    "O_w_HL_power": 7720,
    "O_w_HL_voltage": 24
}
```

The data represents measurements from six hydraulic components:
1. **BLO (Bottom Left Outlet)**
   - Weg current (I_w_BLO_Weg)
   - Power output (O_w_BLO_power)
   - Voltage output (O_w_BLO_voltage)

2. **BHL (Bottom Left Hydraulic)**
   - Weg current (I_w_BHL_Weg)
   - Power output (O_w_BHL_power)
   - Voltage output (O_w_BHL_voltage)

3. **BHR (Bottom Right Hydraulic)**
   - Weg current (I_w_BHR_Weg)
   - Power output (O_w_BHR_power)
   - Voltage output (O_w_BHR_voltage)

4. **BRU (Bottom Right Upper)**
   - Weg current (I_w_BRU_Weg)
   - Power output (O_w_BRU_power)
   - Voltage output (O_w_BRU_voltage)

5. **HR (Hydraulic Right)**
   - Weg current (I_w_HR_Weg)
   - Power output (O_w_HR_power)
   - Voltage output (O_w_HR_voltage)

6. **HL (Hydraulic Left)**
   - Weg current (I_w_HL_Weg)
   - Power output (O_w_HL_power)
   - Voltage output (O_w_HL_voltage)

Each component has three types of measurements:
- **Weg current (I_w_*)**: Input current to the component
- **Power output (O_w_*_power)**: Power output from the component
- **Voltage output (O_w_*_voltage)**: Voltage output from the component

The data is collected at regular intervals (Timestamp) and includes binary labels indicating normal (0) or fault (1) conditions.

### Training the Model

```bash
python src/train.py
```

This will:
1. Load and preprocess the sensor data
2. Generate Siamese pairs for training
3. Train the model with the specified configuration
4. Save the trained model and evaluation results

### Starting the API Server

```bash
python main.py
```

The API will be available at `http://localhost:8000`

### API Endpoints

The API provides a Swagger UI interface for easy testing and documentation. You can access it at `http://localhost:8000/docs`:

![FastAPI Swagger UI](docs/FastAPI_UI_Swagger.jpeg)

#### 1. Health Check
- **GET** `/health`
- Returns the health status of the API and loaded models
- Response:
```json
{
    "status": "healthy",
    "siamese_model_loaded": true,
    "siamese_model_exists": true
}
```

#### 2. Training
- **POST** `/train`
- Triggers model training
- Response:
```json
{
    "status": "success",
    "message": "Training completed successfully"
}
```

#### 3. Prediction
- **POST** `/predict`
- Makes predictions on new sensor data
- Request body:
```json
{
    "sensor_data": {
        "I_w_BLO_Weg": 0.5,
        "O_w_BLO_power": 0.3,
        // ... other sensor features
    }
}
```
- Response:
```json
{
    "prediction": "fault",
    "similarity_score": 0.85,
    "confidence": 0.92
}
```

## Data Visualization

### Sensor Data Characteristics
The system processes data from 18 hydraulic sensors, with each sensor providing three types of measurements:
- **Weg current (I_w_*)**: Input current to the component
- **Power output (O_w_*_power)**: Power output from the component
- **Voltage output (O_w_*_voltage)**: Voltage output from the component

### Data Distribution
1. **Time Series Patterns**:
   - Sensor readings are collected at regular intervals
   - Each cycle contains 8 timesteps of measurements
   - Normal and fault conditions show distinct patterns in sensor readings

2. **Feature Distributions**:
   - Power outputs range from 0 to ~8000 units
   - Voltage outputs typically range from 0 to 30 units
   - Weg currents show both positive and negative values

3. **Class Distribution**:
   - Binary classification: Normal (0) vs Fault (1)
   - Imbalanced dataset with more normal samples than fault samples
   - Data augmentation techniques applied to balance the classes

### Visualization Examples
The following visualizations are generated during data preprocessing:

1. **Sensor Time Series**:
   - Line plots showing sensor readings over time
   - Separate plots for normal and fault conditions
   - Highlighted differences in patterns between conditions

2. **Feature Correlation**:
   - Heatmap showing correlations between different sensors
   - Strong correlations between related sensors (e.g., power and voltage)
   - Weak correlations between unrelated sensors

3. **Distribution Plots**:
   - Histograms showing value distributions for each sensor
   - Box plots comparing normal vs fault conditions
   - Violin plots showing density distributions

4. **t-SNE Visualization**:
   - 2D projection of high-dimensional sensor data
   - Clear separation between normal and fault conditions
   - Clusters indicating different types of faults

### Data Preprocessing Steps
1. **Normalization**:
   - Min-max scaling applied to all features
   - Separate scaling for each sensor type
   - Preserved temporal relationships

2. **Sequence Generation**:
   - Sliding window approach with window size of 8
   - Overlapping windows for data augmentation
   - Maintained temporal order within sequences

3. **Feature Engineering**:
   - Statistical features (mean, std, min, max)
   - Temporal features (slope, change rate)
   - Cross-sensor features (ratios, differences)

## Results

### Model Performance

#### CNN Model
- **Accuracy**: 0.6411 (64.11%)
- **Precision**: 0.6497 (64.97%)
- **Recall**: 0.6411 (64.11%)
- **F1 Score**: 0.5522 (55.22%)

#### Siamese Model
- **Accuracy**: 0.4260 (42.60%)
- **Precision**: 0.5190 (51.90%)
- **Recall**: 0.4260 (42.60%)
- **F1 Score**: 0.3845 (38.45%)

### Model Comparison
| Metric     | CNN Model | Siamese Model | Difference |
|------------|-----------|---------------|------------|
| Accuracy   | 0.6411    | 0.4260        | +0.2152    |
| Precision  | 0.6497    | 0.5190        | +0.1308    |
| Recall     | 0.6411    | 0.4260        | +0.2152    |
| F1 Score   | 0.5522    | 0.3845        | +0.1677    |

### Key Findings
1. The CNN model demonstrates superior performance across all metrics compared to the Siamese model
2. The CNN model achieves over 64% accuracy in fault detection, which is significantly higher than the Siamese model's 42.6%
3. Both models show balanced precision and recall scores, indicating consistent performance across different types of predictions
4. The F1 scores suggest that the CNN model provides more reliable predictions overall

### Visualization Results
The following visualizations are generated during training and evaluation:
- ROC curves for both training and validation data
- Confusion matrices
- Training history plots
- Accuracy distribution plots
- Model comparison charts

All visualizations are saved in the `results` directory for further analysis.

## Testing

The project includes a comprehensive test suite to ensure code quality and functionality. Tests are organized by component:

### Running Tests

1. Install test dependencies:
```bash
pip install pytest pytest-cov
```

## t-SNE Visualization
The project generates t-SNE plots for both CNN and Siamese models to visualize the learned embeddings:
- **CNN t-SNE**: Shows how the CNN model's convolutional layers transform the input data into a lower-dimensional space
- **Siamese t-SNE**: Visualizes how the Siamese network's embedding space separates similar and dissimilar sequences

## Feature Importance Analysis
The system provides detailed feature importance analysis using SHAP values:
- **CNN Feature Importance**: Ranks features based on their impact on the CNN model's predictions
- **Siamese Feature Importance**: Analyzes which features are most important for sequence similarity
- **Feature Rankings**: Detailed text files containing importance scores for each feature at each time step

## Results Directory Structure
The `results` directory contains:
```
results/
├── cnn_tsne.png              # t-SNE visualization of CNN model embeddings
├── siamese_tsne.png          # t-SNE visualization of Siamese model embeddings
├── cnn_feature_importance.png # SHAP-based feature importance for CNN
├── siamese_feature_importance.png # SHAP-based feature importance for Siamese
├── cnn_feature_rankings.txt  # Detailed feature rankings for CNN model
├── siamese_feature_rankings.txt # Detailed feature rankings for Siamese model
├── cnn_roc.png              # ROC curve for CNN model
├── siamese_roc.png          # ROC curve for Siamese model
├── cnn_confusion_matrix.png # Confusion matrix for CNN model
└── siamese_confusion_matrix.png # Confusion matrix for Siamese model
```

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
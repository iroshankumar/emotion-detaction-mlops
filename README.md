# Emotion Detection MLOps Project

## 📌 Project Overview
This project is an end-to-end MLOps implementation for **Emotion Detection** using the **FER2013 (Facial Expression Recognition 2013)** dataset. The goal is to classify human facial expressions into 7 categories: `Angry`, `Disgust`, `Fear`, `Happy`, `Sad`, `Surprise`, and `Neutral`.

The project follows a modular coding structure, integrating industry-standard MLOps tools like **DVC (Data Version Control)** for pipeline management and **MLflow** via **DAGsHub** for experiment tracking.

---

## 🚀 Key Features
- **Modular Pipeline**: Separate components for Data Ingestion, Data Transformation, and Model Training.
- **Deep Learning Model**: CNN-based architecture implemented using TensorFlow/Keras.
- **Experiment Tracking**: Integrated with MLflow and DAGsHub to log parameters, metrics, and model versions.
- **Pipeline Orchestration**: DVC-managed training pipeline for reproducibility.
- **Web Application**: A Flask-based web interface to upload images and get real-time emotion predictions.
- **Configuration Management**: Centralized `params.yaml` for easy tuning of hyperparameters and paths.

---

## 🛠️ Tech Stack
- **Languages**: Python
- **Libraries**: TensorFlow, Keras, OpenCV, Pandas, NumPy, Scikit-learn
- **Frameworks**: Flask (Web App)
- **MLOps**: DVC, MLflow, DAGsHub
- **Logging & Exceptions**: Custom implementation for robust debugging.

---

## 📂 Project Structure
```text
.
├── artifacts/              # Model artifacts and saved models
├── config/                 # Configuration files
├── data/                   # Dataset (Raw, Processed, etc.)
│   └── raw/                # FER2013 images (Train, Test, Val folders)
├── logs/                   # Project execution logs
├── notebooks/              # Jupyter notebooks for experimentation
├── reports/                # Training metrics and history (JSON)
├── src/                    # Source code
│   ├── components/         # Core logic (Ingestion, Transformation, Training)
│   ├── config/             # Configuration management logic
│   ├── entity/             # Configuration entities
│   ├── pipeline/           # Training and Prediction pipelines
│   ├── utils.py            # Utility functions
│   ├── logger.py           # Logging setup
│   └── exception.py        # Custom exception handling
├── static/                 # Web app static files (CSS, Uploads)
├── templates/              # Flask HTML templates
├── app.py                  # Flask Web Entry Point
├── dvc.yaml                # DVC Pipeline definition
├── main.py                 # CLI Entry Point
├── params.yaml             # Hyperparameters and folder paths
├── requirements.txt        # Project dependencies
└── setup.py                # Package configuration
```

---

## 🛠️ Installation & Setup

### 1. Clone the repository
```bash
git clone <repository-url>
cd emotion-detection-mlops
```

### 2. Create and activate a Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## ⚙️ MLOps Workflow

### 1. Data Pipeline (DVC)
The project uses DVC to manage the pipeline. Currently, it includes a `train` stage.
To run the training pipeline:
```bash
dvc repro
```

### 2. Experiment Tracking (MLflow + DAGsHub)
The training process is automatically tracked.
- **Parameters logged**: `learning_rate`, `batch_size`, `epochs`, `image_size`.
- **Metrics logged**: `val_loss`, `val_accuracy`.
- **Artifacts**: The trained model (.keras) and metrics are saved.

To view the experiments, visit your linked **DAGsHub** repository.

---

## 🧠 Model Architecture
The model is a Convolutional Neural Network (CNN) consisting of:
- **3 Conv2D layers** with ReLU activation and MaxPooling2D.
- **Flatten layer** to convert 2D features to 1D.
- **Dense layer (128 units)** with ReLU and Dropout (0.5).
- **Softmax Output layer** for 7-class classification.

---

## 🌐 Web Interface
To run the web application for inference:
```bash
python app.py
```
Visit `http://127.0.0.1:5000` in your browser. You can upload an image, and it will display the predicted emotion and confidence level.

---

## 📝 Configuration (`params.yaml`)
You can tweak the training process without changing the code:
```yaml
training:
  epochs: 5
  batch_size: 32
  learning_rate: 0.001
model:
  num_classes: 7
```

---

## 📈 Recent Progress
- [x] Initial Project Structure setup.
- [x] Modular implementation of Data Ingestion and Transformation.
- [x] Model Trainer with MLflow integration.
- [x] DVC pipeline integration for the training stage.
- [x] Flask-based prediction interface.
- [x] Integrated DAGsHub for remote experiment tracking.

---

## 🤝 Acknowledgments
- **Dataset**: FER2013 Dataset from Kaggle.
- **Inspiration**: MLOps best practices and modular programming.

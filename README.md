# Face Anti-Spoofing Binary Classifier

This repository contains an end-to-end implementation of a Face Anti-Spoofing (FAS) binary classifier built with PyTorch. It fine-tunes a powerful ResNet-50 model to classify images as either a `REAL FACE` or a `SPOOF ATTACK`.

## Project Structure
```
.
├── data/                  # Mock or real dataset (ignored in git)
├── models/                # Saved trained model checkpoints
├── plots/                 # Metrics and evaluation plots
├── src/
│   ├── dataset.py         # PyTorch Dataset and augmentations
│   ├── model.py           # ResNet-50 Binary Classifier architecture
│   ├── train.py           # Training loop and loss tracking
│   ├── evaluate.py        # Model evaluation and metrics generation
│   └── infer.py           # Single-image inference, Grad-CAM, and Webcam Demo
├── requirements.txt       # Python dependencies
├── report.md              # Technical report summarizing approach and results
├── slides.md              # Presentation slides
└── README.md              # This instructions file
```

## Setup Instructions

1. **Install Requirements:**
   Make sure you have Python 3.8+ installed. Install the dependencies using:
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare the Data:**
   You can place your dataset inside the `data/` directory with the following structure:
   ```
   data/
   ├── train/
   │   ├── 0_real/
   │   └── 1_spoof/
   ├── val/
   │   ├── 0_real/
   │   └── 1_spoof/
   └── test/
       ├── 0_real/
       └── 1_spoof/
   ```
   *Alternatively, generate a dummy dataset for quick testing:*
   ```bash
   PYTHONPATH=src python -c "from src.dataset import generate_mock_dataset; generate_mock_dataset('data')"
   ```

## Usage Instructions

### 1. Training the Model
To start training the ResNet-50 model on your dataset, run:
```bash
PYTHONPATH=src python src/train.py
```
This will train the model for 3 epochs (can be adjusted inside `train.py`), save the best model to `models/best_fas_model.pth`, and output loss/accuracy plots to the `plots/` directory.

### 2. Evaluating the Model
To evaluate the model on your test dataset and generate a Confusion Matrix and ROC Curve:
```bash
PYTHONPATH=src python src/evaluate.py
```
Outputs will be saved in the `plots/` directory.

### 3. Inference & Webcam Demo
The `infer.py` script provides two exciting modes of operation: single image evaluation with Grad-CAM visualization, and a real-time webcam demo.

**Single Image Mode (with Grad-CAM):**
```bash
PYTHONPATH=src python src/infer.py --mode image --image path/to/your/image.jpg
```
This evaluates the image, prints the prediction probability, and saves a `gradcam_result.jpg` visualizing the model's attention.

**Webcam Demo Mode:**
```bash
PYTHONPATH=src python src/infer.py --mode webcam
```
This uses OpenCV to capture your webcam, detects faces using a Haarcascade classifier, and runs real-time anti-spoofing predictions displaying bounding boxes and labels on the feed. Press `q` to quit the feed.

## Results
- The framework successfully integrates transfer learning through ResNet-50, yielding fast training capabilities and high accuracy.
- Training supports device-agnostic execution dynamically mapping to GPU (`cuda`) if available, or seamlessly falling back to `cpu` for compatibility.
- Generated plots and visual attention maps (Grad-CAM) add profound interpretability to the anti-spoofing mechanism.

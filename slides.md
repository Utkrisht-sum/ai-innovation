# Face Anti-Spoofing: Deep Learning Binary Classifier
** Semester End Year Project Presentation**

---

## 1. Project Motivation
* **The Problem:** Biometric authentication systems (e.g., facial recognition) are highly susceptible to Presentation Attacks (PA).
* **Spoofing Techniques:** Printed photographs, replay attacks on digital screens, and 3D masks.
* **The Goal:** Develop a fast, effective, and demonstrable **Face Anti-Spoofing (FAS)** binary classifier to detect "Real" vs "Spoof" faces.

---

## 2. Project Scope & Strategy
* **Demonstration over Research:** Prioritized a robust end-to-end working pipeline within a strict 24-hour development cycle.
* **Lightweight Approach:** Train on small, balanced subsets or extracted frames rather than massive video datasets.
* **Key Deliverables:**
  * Complete Training & Evaluation Pipeline
  * Real-Time Webcam Demo
  * Model Interpretability (Grad-CAM)
  * Device-Agnostic Execution (CPU or GPU)

---

## 3. Model Architecture
* **Chosen Model:** `ResNet-50`
  * Strong capacity (~25M parameters) with excellent transfer learning properties.
  * Faster and more reliable than complex Transformers or CNN-LSTM pipelines for limited training time.
* **Modifications:**
  * Loaded with Pretrained ImageNet weights.
  * Replaced the final Fully Connected (`fc`) layer for 2-class output (Binary Classifier).

---

## 4. Training Pipeline
* **Data Processing:**
  * Loaded via PyTorch `Dataset` & `DataLoader`.
  * Augmentations applied: Random Horizontal Flips, Color Jitter, 224x224 Resize.
* **Hyperparameters:**
  * Loss: CrossEntropyLoss
  * Optimizer: Adam (Learning Rate: 1e-4)
  * Epochs: 3–5 (Optimized for rapid demonstration and avoiding overfitting on small samples).
* **Checkpointing:** Automatically saves the model with the highest validation accuracy.

---

## 5. Evaluation & Metrics
* **Testing Protocol:** Evaluated on an unseen test split.
* **Visualized Metrics Generated:**
  * **Accuracy/Loss Curves:** Monitored over the training duration.
  * **Confusion Matrix:** Tracks True Positives (Real) and False Positives (Spoof).
  * **ROC Curve & AUC Score:** Measures the separability between the genuine and attack distributions.

---

## 6. High-Impact Feature: Live Demo
* **Real-Time Webcam Inference:**
  * Utilizes `OpenCV` with Haar Cascade face detection.
  * Captures live frames, dynamically crops the face, and performs ResNet-50 inference.
  * Displays color-coded bounding boxes:
    * 🟢 **REAL FACE**
    * 🔴 **SPOOF ATTACK**

---

## 7. High-Impact Feature: Grad-CAM
* **Interpretability (Why did the model choose that?):**
  * Implemented Gradient-weighted Class Activation Mapping (Grad-CAM).
  * Attaches hooks to the final convolutional layer of ResNet-50.
  * **Result:** Generates a heatmap overlay highlighting the specific facial features (or screen artifacts) the model focused on to make its spoofing prediction.

---

## 8. Conclusion
* Successfully developed a highly demonstrable, end-to-end FAS pipeline.
* Struck an effective balance between modern Deep Learning architectures (ResNet-50) and practical execution constraints.
* The webcam interface and Grad-CAM outputs create an engaging, visually compelling final year project suitable for interactive demonstration.

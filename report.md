# Face Anti-Spoofing Deep Learning Binary Classifier
## Final Year Project Technical Report

### 1. Executive Summary
This project aims to develop a robust, end-to-end Face Anti-Spoofing (FAS) binary classifier to differentiate between genuine faces ("Real") and presentation attacks ("Spoof"). Presentation attacks—such as printed photos or digital replays—pose severe security vulnerabilities to biometric systems.

Given the constraints of a 24-hour final year project cycle, the core strategy was to prioritize a working, demonstrable pipeline over exhaustive research-scale training. We utilized a transfer learning approach leveraging a pretrained ResNet-50 architecture. The final implementation includes an end-to-end training and evaluation pipeline, coupled with a real-time webcam demonstration and model interpretability via Grad-CAM.

### 2. Dataset Strategy and Preprocessing
To facilitate rapid iteration and a functional end-to-end pipeline, the project supports lightweight, pre-cropped facial image datasets. The data loader (`FASDataset`) leverages PyTorch's `Dataset` class, organized into distinct `train`, `val`, and `test` splits, with categorical subdirectories for `0_real` and `1_spoof`.

**Data Augmentation:**
To mitigate overfitting on smaller datasets and improve model generalization, we employ standard `torchvision` transformations during the training phase, including:
- Random Horizontal Flips
- Color Jitter (Brightness & Contrast perturbations)
- Image Resizing to 224x224 (compatible with ResNet input layers)
- ImageNet Normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

*Note:* A synthetic mock-dataset generation script is included to immediately verify the pipeline out-of-the-box without requiring gigabytes of raw video downloads.

### 3. Model Architecture
The classifier utilizes the **ResNet-50** deep convolutional neural network. ResNet-50 was selected over alternatives like Transformers or heavy 3D-CNNs for its established balance of parameter efficiency (approx. ~25M), representation capacity, and excellent transfer learning performance on image classification tasks.

**Architecture Details:**
- **Base Model:** Pretrained on ImageNet (via `ResNet50_Weights.DEFAULT`), ensuring robust foundational feature extractors (edges, textures, shapes).
- **Classification Head:** The final fully connected (`fc`) layer is replaced with a linear layer mapping the latent features to 2 output classes (Real, Spoof).
- **Execution:** The implementation is strictly device-agnostic, seamlessly transitioning to hardware acceleration (`cuda`) if available or falling back to CPU, satisfying accessibility and demonstration constraints.

### 4. Training and Evaluation Pipeline
**Training Protocol:**
The model trains using a Cross-Entropy Loss function optimized via the Adam optimizer with a learning rate of `1e-4`. Training executes for a deliberately concise 3–5 epochs, perfectly suited for rapid demonstration. Validation accuracy is monitored per epoch, and the best-performing model state is checkpointed (`models/best_fas_model.pth`).

**Evaluation Metrics:**
The testing framework computes several robust metrics to evaluate performance:
- **Global Accuracy:** Baseline performance metric.
- **Confusion Matrix:** Provides transparent visibility into True Positives/Negatives and False Positives/Negatives (APCER/BPCER equivalents).
- **Receiver Operating Characteristic (ROC) & AUC:** Validates the model's distinct separability between Real and Spoof distributions.

### 5. Demonstrations and Interpretability (High Impact Features)
This project focuses heavily on interactive demonstration, encapsulated in `infer.py`:

**A. Real-Time Webcam Inference:**
We integrate OpenCV's highly efficient Haar Cascades for real-time face localization on webcam streams. Detected faces are dynamically cropped, preprocessed, and passed to the inference model. Bounding boxes are drawn onto the live feed, labeled distinctively as `REAL FACE` (Green) or `SPOOF ATTACK` (Red) accompanied by the softmax probability.

**B. Grad-CAM (Gradient-weighted Class Activation Mapping):**
To provide interpretability into *why* the model makes a decision, we implemented Grad-CAM. By attaching forward and backward hooks to the final convolutional layer of the ResNet backbone (`layer4[-1].conv3`), the model highlights the spatial regions of the image that contributed most heavily to the positive prediction class. This visual heatmap output (`gradcam_result.jpg`) strongly reinforces trust in the model's behavior.

### 6. Conclusion
This project successfully fulfills the criteria of a high-quality demonstration-focused FAS pipeline. The utilization of transfer learning yields rapid convergence, while the integrated real-time webcam module and Grad-CAM interpretability provide compelling visual evidence of the model's efficacy, ultimately achieving a balance of academic rigor and functional demonstration.

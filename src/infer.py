import cv2
import torch
import numpy as np
import argparse
import os
from PIL import Image
import torchvision.transforms.functional as F
from model import FASModel, get_device
from dataset import get_transforms

def get_gradcam(model, image_tensor, target_class):
    """
    Basic Grad-CAM implementation for ResNet50
    """
    model.eval()

    # Hooks to store gradients and activations
    activations = None
    gradients = None

    def forward_hook(module, input, output):
        nonlocal activations
        activations = output

    def backward_hook(module, grad_in, grad_out):
        nonlocal gradients
        gradients = grad_out[0]

    # Register hooks on the last convolutional layer (layer4 in ResNet)
    target_layer = model.model.layer4[-1].conv3
    handle_forward = target_layer.register_forward_hook(forward_hook)
    handle_backward = target_layer.register_full_backward_hook(backward_hook)

    # Forward pass
    output = model(image_tensor)

    # Backward pass
    model.zero_grad()
    loss = output[0, target_class]
    loss.backward()

    # Calculate Grad-CAM
    weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
    cam = torch.sum(weights * activations, dim=1).squeeze()
    cam = torch.relu(cam) # ReLU
    cam = cam - torch.min(cam)
    if torch.max(cam) > 0:
        cam = cam / torch.max(cam) # Normalize to 0-1

    handle_forward.remove()
    handle_backward.remove()

    return cam.cpu().detach().numpy()

def overlay_gradcam(image, cam):
    cam = cv2.resize(cam, (image.shape[1], image.shape[0]))
    cam = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    result = heatmap * 0.4 + image * 0.6
    return np.uint8(result)

def infer_single_image(image_path, model_path='models/best_model.pt'):
    device = get_device()
    model = FASModel(pretrained=False)
    if not os.path.exists(model_path):
        print(f"Error: Model {model_path} not found.")
        return
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    transform = get_transforms(is_train=False)

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    input_tensor = transform(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        prob_spoof = probs[0][1].item()
        pred_class = 1 if prob_spoof > 0.5 else 0

    label = "SPOOF ATTACK" if pred_class == 1 else "REAL FACE"
    color = (0, 0, 255) if pred_class == 1 else (0, 255, 0)

    # Generate Grad-CAM for the predicted class
    # To use backward hooks we need requires_grad=True
    input_tensor_grad = input_tensor.clone().detach().requires_grad_(True)
    cam = get_gradcam(model, input_tensor_grad, pred_class)

    # Overlay and display
    image_resized = cv2.resize(image, (224, 224))
    cam_overlay = overlay_gradcam(image_resized, cam)

    cv2.putText(cam_overlay, f"{label} ({prob_spoof:.2f})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imwrite("gradcam_result.jpg", cam_overlay)
    print(f"Prediction: {label} (Spoof Prob: {prob_spoof:.4f}). Result saved to gradcam_result.jpg")

def run_webcam(model_path='models/best_model.pt'):
    device = get_device()
    model = FASModel(pretrained=False)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except FileNotFoundError:
        print(f"Error: Model {model_path} not found. Please train first.")
        return

    model.to(device)
    model.eval()
    transform = get_transforms(is_train=False)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Webcam started. Press 'q' to quit.")

    # Basic face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            # Crop face
            face_img = frame[y:y+h, x:x+w]
            face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            pil_face = Image.fromarray(face_rgb)

            input_tensor = transform(pil_face).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(input_tensor)
                probs = torch.softmax(output, dim=1)
                prob_spoof = probs[0][1].item()
                pred_class = 1 if prob_spoof > 0.5 else 0

            label = "SPOOF ATTACK" if pred_class == 1 else "REAL FACE"
            color = (0, 0, 255) if pred_class == 1 else (0, 255, 0)

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"{label} ({prob_spoof:.2f})", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow('Face Anti-Spoofing Demo', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    import os
    parser = argparse.ArgumentParser(description="FAS Inference")
    parser.add_argument('--mode', type=str, choices=['image', 'webcam'], default='image', help="Inference mode")
    parser.add_argument('--image', type=str, help="Path to input image")
    args = parser.parse_args()

    if args.mode == 'image':
        if not args.image:
            print("Please provide --image path for image mode")
        else:
            infer_single_image(args.image)
    else:
        run_webcam()

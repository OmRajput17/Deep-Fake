"""
Explainable AI — Grad-CAM Visualization for Deepfake Detection.

Generates heatmaps showing which facial regions the model focuses on
to distinguish real from fake faces.

Supports:
  - Single image analysis
  - Video analysis (annotated with heatmap overlay)
  - Batch image analysis (saves heatmaps to output folder)

Usage:
    python explain.py --image_path ./test_face.jpg --model_path ./output/ff_effnet_v1/best.pth
    python explain.py --video_path ./test_video.mp4 --model_path ./output/ff_effnet_v1/best.pth
    python explain.py --image_dir ./processed_faces/original/000 --model_path ./output/ff_effnet_v1/best.pth
"""
import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
from PIL import Image
from tqdm import tqdm
from facenet_pytorch import MTCNN

from network.models import model_selection
from dataset.transform import data_transforms


class GradCAM:
    """
    Grad-CAM: Visual Explanations from Deep Networks.
    Targets the last convolutional layer of EfficientNet-B4 backbone.
    """

    def __init__(self, model, target_layer=None):
        self.model = model
        self.model.eval()
        self.gradients = None
        self.activations = None

        # Default: last conv layer in EfficientNet-B4 features
        if target_layer is None:
            target_layer = model.backbone.features[-1]

        self.target_layer = target_layer

        # Register hooks
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class=None):
        """
        Generate Grad-CAM heatmap.

        Args:
            input_tensor: preprocessed image tensor (1, 3, H, W)
            target_class: class to explain (None = predicted class)

        Returns:
            heatmap: numpy array (H, W) in [0, 1]
            pred_class: predicted class index
            confidence: prediction confidence
        """
        # Forward pass
        output = self.model(input_tensor)
        probs = F.softmax(output, dim=1)
        pred_class = output.argmax(dim=1).item()
        confidence = probs[0, pred_class].item()

        if target_class is None:
            target_class = pred_class

        # Backward pass for target class
        self.model.zero_grad()
        output[0, target_class].backward()

        # Compute Grad-CAM
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # Global avg pool
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)  # Only positive contributions

        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())

        return cam, pred_class, confidence


def overlay_heatmap(image, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """Overlay Grad-CAM heatmap on original image."""
    h, w = image.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_colored = cv2.applyColorMap(
        (heatmap_resized * 255).astype(np.uint8), colormap
    )
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
    return overlay, heatmap_colored


def create_explanation_panel(image, heatmap, pred_class, confidence):
    """
    Create a side-by-side explanation panel:
    [Original | Heatmap | Overlay]
    """
    h, w = image.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_colored = cv2.applyColorMap(
        (heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    overlay = cv2.addWeighted(image, 0.5, heatmap_colored, 0.5, 0)

    # Labels
    label = 'FAKE' if pred_class == 1 else 'REAL'
    color = (0, 0, 255) if pred_class == 1 else (0, 255, 0)

    # Add text labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, 'Original', (10, 30), font, 0.7, (255, 255, 255), 2)
    cv2.putText(heatmap_colored, 'Grad-CAM', (10, 30), font, 0.7, (255, 255, 255), 2)
    cv2.putText(overlay, f'{label} ({confidence:.1%})', (10, 30), font, 0.7, color, 2)

    panel = np.hstack([image, heatmap_colored, overlay])
    return panel


def explain_image(image_path, gradcam, preprocess, device):
    """Generate Grad-CAM explanation for a single image."""
    img = Image.open(image_path).convert('RGB')
    input_tensor = preprocess(img).unsqueeze(0).to(device)

    # Enable gradients for Grad-CAM
    input_tensor.requires_grad_(True)

    heatmap, pred_class, confidence = gradcam.generate(input_tensor)

    # Load original for visualization
    original = cv2.imread(image_path)
    original = cv2.resize(original, (380, 380))

    panel = create_explanation_panel(original, heatmap, pred_class, confidence)
    return panel, pred_class, confidence


def explain_video(video_path, model_path, output_path='./explanations'):
    """Generate Grad-CAM annotated video."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model_selection('efficientnet_b4', num_out_classes=2, dropout=0.5)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    gradcam = GradCAM(model)
    mtcnn = MTCNN(keep_all=False, select_largest=True, device=device)
    preprocess = data_transforms['test']
    softmax = nn.Softmax(dim=1)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs(output_path, exist_ok=True)
    out_file = os.path.join(output_path,
                            os.path.basename(video_path).rsplit('.', 1)[0] + '_gradcam.avi')
    writer = cv2.VideoWriter(out_file, cv2.VideoWriter_fourcc(*'MJPG'), fps, (w, h))

    real_count = fake_count = 0
    pbar = tqdm(total=total, desc="Generating Grad-CAM")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        pbar.update(1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        boxes, probs = mtcnn.detect(pil_img)

        if boxes is not None and len(boxes) > 0:
            best = probs.argmax()
            x1, y1, x2, y2 = [int(b) for b in boxes[best]]

            # Expand crop
            bw, bh = x2 - x1, y2 - y1
            size = int(max(bw, bh) * 1.3)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            nx1 = max(cx - size // 2, 0)
            ny1 = max(cy - size // 2, 0)
            size = min(w - nx1, size)
            size = min(h - ny1, size)

            crop = frame[ny1:ny1 + size, nx1:nx1 + size]
            if crop.size > 0:
                img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                pil_crop = Image.fromarray(img)
                tensor = preprocess(pil_crop).unsqueeze(0).to(device)
                tensor.requires_grad_(True)

                heatmap, pred_class, confidence = gradcam.generate(tensor)

                # Overlay heatmap on the face region in the original frame
                heatmap_resized = cv2.resize(heatmap, (size, size))
                heatmap_colored = cv2.applyColorMap(
                    (heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET
                )
                frame[ny1:ny1 + size, nx1:nx1 + size] = cv2.addWeighted(
                    crop, 0.6, heatmap_colored, 0.4, 0
                )

                label = 'FAKE' if pred_class == 1 else 'REAL'
                color = (0, 0, 255) if pred_class == 1 else (0, 255, 0)
                if pred_class == 1:
                    fake_count += 1
                else:
                    real_count += 1

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f'{label} ({confidence:.2f})', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        writer.write(frame)

    cap.release()
    writer.release()
    pbar.close()

    total_f = real_count + fake_count
    if total_f > 0:
        verdict = 'FAKE' if fake_count > real_count else 'REAL'
        print(f'\n🎯 VERDICT: {verdict}')
        print(f'   Real: {real_count} ({real_count / total_f * 100:.1f}%)')
        print(f'   Fake: {fake_count} ({fake_count / total_f * 100:.1f}%)')
    print(f'   Output: {out_file}')


def explain_batch(image_dir, model_path, output_path='./explanations'):
    """Generate Grad-CAM explanations for a folder of images."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model_selection('efficientnet_b4', num_out_classes=2, dropout=0.5)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    gradcam = GradCAM(model)
    preprocess = data_transforms['test']

    os.makedirs(output_path, exist_ok=True)

    exts = ('.jpg', '.jpeg', '.png', '.bmp')
    images = [f for f in os.listdir(image_dir)
              if f.lower().endswith(exts)]

    print(f'\n🔍 Analyzing {len(images)} images from {image_dir}...\n')

    real_count = fake_count = 0
    for img_name in tqdm(images, desc="Generating explanations"):
        img_path = os.path.join(image_dir, img_name)
        panel, pred_class, confidence = explain_image(img_path, gradcam, preprocess, device)

        label = 'FAKE' if pred_class == 1 else 'REAL'
        if pred_class == 1:
            fake_count += 1
        else:
            real_count += 1

        out_name = f'{label}_{confidence:.2f}_{img_name}'
        cv2.imwrite(os.path.join(output_path, out_name), panel)

    print(f'\n📊 Results: Real={real_count} | Fake={fake_count}')
    print(f'   Explanations saved to: {output_path}/')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Explainable AI — Grad-CAM for Deepfake Detection')
    parser.add_argument('--model_path', '-m', required=True,
                        help='Path to trained model weights')
    parser.add_argument('--image_path', '-i', default=None,
                        help='Path to a single image')
    parser.add_argument('--video_path', '-v', default=None,
                        help='Path to a video file')
    parser.add_argument('--image_dir', '-d', default=None,
                        help='Path to a directory of images')
    parser.add_argument('--output_path', '-o', default='./explanations',
                        help='Output directory for results')
    args = parser.parse_args()

    if args.video_path:
        explain_video(args.video_path, args.model_path, args.output_path)

    elif args.image_dir:
        explain_batch(args.image_dir, args.model_path, args.output_path)

    elif args.image_path:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model_selection('efficientnet_b4', num_out_classes=2, dropout=0.5)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model = model.to(device)
        model.eval()

        gradcam = GradCAM(model)
        preprocess = data_transforms['test']

        os.makedirs(args.output_path, exist_ok=True)

        panel, pred_class, confidence = explain_image(
            args.image_path, gradcam, preprocess, device)

        label = 'FAKE' if pred_class == 1 else 'REAL'
        out_file = os.path.join(args.output_path,
                                f'gradcam_{label}_{confidence:.2f}.png')
        cv2.imwrite(out_file, panel)
        print(f'\n🎯 Prediction: {label} ({confidence:.1%})')
        print(f'   Explanation saved: {out_file}')
    else:
        print('Please provide --image_path, --video_path, or --image_dir')

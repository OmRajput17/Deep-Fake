# 🔍 DeepFake Detection — Upgraded Pipeline

A state-of-the-art deepfake detection system using **EfficientNet-B4** with ImageNet transfer learning, **MTCNN** face detection, and **Grad-CAM** explainability.

## 🏆 Results

| Metric | Score |
|--------|-------|
| **Accuracy** | 99.1% |
| **Precision** | 99.95% |
| **Recall** | 98.96% |
| **F1 Score** | 99.45% |

Trained on FaceForensics++ (C23 compression) with 12% data sampling (~840 videos, ~44K face images).

## ⚡ Key Features

| Feature | Implementation |
|---------|---------------|
| Backbone | EfficientNet-B4 (ImageNet pretrained) |
| Face Detection | MTCNN (multi-task CNN) |
| Class Balance | WeightedRandomSampler |
| LR Schedule | CosineAnnealingWarmRestarts |
| Training Speed | Mixed Precision (AMP) |
| Augmentation | Flip + Rotation + ColorJitter + GaussianBlur + RandomErasing + Grayscale |
| Overfitting Guard | Early stopping (patience=5) |
| Explainability | Grad-CAM heatmap visualization |
| Test Metrics | Accuracy + Precision + Recall + F1 + Confusion Matrix |

## 📂 Project Structure

```
DeepFake/
├── network/
│   ├── __init__.py
│   └── models.py              ← EfficientNet-B4 + ImageNet pretrained
├── dataset/
│   ├── __init__.py
│   ├── transform.py           ← 380x380, ImageNet norm, enhanced augmentation
│   └── mydataset.py           ← Dataset loader + WeightedRandomSampler support
├── preprocess_dataset.py      ← MTCNN face extraction (GPU + parallel workers)
├── train_CNN.py               ← AMP + class balance + cosine annealing + early stop
├── test_CNN.py                ← Full metrics (Acc/Prec/Rec/F1/Confusion Matrix)
├── predict.py                 ← Video-level prediction with annotated output
└── explain.py                 ← Grad-CAM explainable AI visualization
```

## 🚀 Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install torch torchvision opencv-python numpy pillow tqdm facenet-pytorch
```

> **Note:** Install PyTorch with CUDA for GPU acceleration:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
> ```

## 📋 Usage

### 1. Preprocess — Extract faces from videos
```bash
python preprocess_dataset.py -d "/path/to/FaceForensics++_C23" -o "./processed_faces" --sample_ratio 0.12 --num_workers 8
```

### 2. Train
```bash
python train_CNN.py --name ff_effnet_v1 --batch_size 8 --epoches 20 --lr 0.001
```

### 3. Test
```bash
python test_CNN.py --test_list ./data_list/ff_all_test.txt --model_path ./output/ff_effnet_v1/best.pth
```

### 4. Predict on video
```bash
python predict.py -i ./video.mp4 -m ./output/ff_effnet_v1/best.pth
```

### 5. Explainable AI (Grad-CAM)
```bash
# Single image
python explain.py -i ./face.png -m ./output/ff_effnet_v1/best.pth

# Batch of images
python explain.py -d ./processed_faces/Deepfakes/001_870 -m ./output/ff_effnet_v1/best.pth

# Video with heatmap overlay
python explain.py -v ./video.mp4 -m ./output/ff_effnet_v1/best.pth
```

## 🧠 How It Works

1. **MTCNN** detects and crops faces from video frames
2. **EfficientNet-B4** (pretrained on ImageNet) classifies each face as REAL or FAKE
3. Training uses a 2-phase approach:
   - Phase 1 (epochs 1-2): Frozen backbone, train classifier head only
   - Phase 2 (epochs 3+): Full fine-tuning with differential learning rates
4. **Grad-CAM** generates heatmaps showing which facial regions influenced the decision

## 📊 Grad-CAM Examples

Each explanation panel shows: **Original → Heatmap → Overlay with prediction**

- **Real faces**: Model focuses on natural skin texture (nose area)
- **Fake faces**: Model highlights face-swap boundaries (eyes, jawline, mouth edges)

## 🛠️ Requirements

- Python 3.10+
- PyTorch 2.0+ (with CUDA for GPU)
- NVIDIA GPU recommended (tested on RTX 3050)

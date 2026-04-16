"""
Enhanced transforms for EfficientNet-B4.
- Input: 380x380 (EfficientNet-B4 native resolution)
- Normalization: ImageNet mean/std
- Extra augment: GaussianBlur, RandomErasing, RandomGrayscale, JPEG compression
"""
from torchvision import transforms
from PIL import Image
import io
import random


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
INPUT_SIZE = 380  # EfficientNet-B4 native resolution


class JPEGCompression:
    """Simulate JPEG compression artifacts (phone cameras, social media)."""
    def __init__(self, quality_range=(30, 95), p=0.5):
        self.quality_range = quality_range
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            quality = random.randint(*self.quality_range)
            buf = io.BytesIO()
            img.save(buf, format='JPEG', quality=quality)
            buf.seek(0)
            img = Image.open(buf).convert('RGB')
        return img


class RandomDownscale:
    """Simulate low-resolution phone/webcam capture."""
    def __init__(self, scale_range=(0.5, 1.0), p=0.3):
        self.scale_range = scale_range
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            w, h = img.size
            scale = random.uniform(*self.scale_range)
            new_w, new_h = int(w * scale), int(h * scale)
            if new_w > 10 and new_h > 10:
                img = img.resize((new_w, new_h), Image.BILINEAR)
                img = img.resize((w, h), Image.BILINEAR)
        return img


data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.4, contrast=0.4,
                               saturation=0.3, hue=0.08),
        transforms.RandomGrayscale(p=0.05),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        JPEGCompression(quality_range=(30, 95), p=0.5),
        RandomDownscale(scale_range=(0.5, 1.0), p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
    ]),
    'val': transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]),
    'test': transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]),
}

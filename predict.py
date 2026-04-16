"""
PHASE 4: Run trained model on new video files.
Outputs annotated video + verdict.

Includes confidence threshold to reduce false positives on real-world videos.

Usage:
    python predict.py --video_path ./test_video.mp4 --model_path ./output/ff_effnet_v1/best.pth
    python predict.py -i ./test_video.mp4 -m ./output/ff_effnet_v1/best.pth --threshold 0.75
"""
import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from PIL import Image as pil_image
from tqdm import tqdm
from facenet_pytorch import MTCNN

from network.models import model_selection
from dataset.transform import data_transforms


def predict_video(video_path, model_path, output_path='./predictions',
                  fake_threshold=0.70, verdict_ratio=0.6):
    """
    Analyze video for deepfake content.

    Args:
        fake_threshold: Minimum confidence to classify a frame as FAKE (default 0.70).
                        Below this → classified as REAL (avoids borderline false positives).
        verdict_ratio:  Fraction of frames that must be FAKE for the video verdict
                        to be FAKE (default 0.6 = 60%). Reduces false positives.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = model_selection('efficientnet_b4', num_out_classes=2, dropout=0.5)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # MTCNN face detector
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
                            os.path.basename(video_path).rsplit('.', 1)[0] + '_result.avi')
    writer = cv2.VideoWriter(out_file, cv2.VideoWriter_fourcc(*'MJPG'), fps, (w, h))

    real_count = fake_count = uncertain_count = 0
    fake_confidences = []
    pbar = tqdm(total=total, desc="Analyzing")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        pbar.update(1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = pil_image.fromarray(rgb)

        boxes, probs = mtcnn.detect(pil_img)

        if boxes is not None and len(boxes) > 0:
            best = probs.argmax()
            x1, y1, x2, y2 = [int(b) for b in boxes[best]]

            # Expand crop
            bw, bh = x2 - x1, y2 - y1
            size = int(max(bw, bh) * 1.3)
            cx, cy = (x1+x2)//2, (y1+y2)//2
            nx1 = max(cx - size//2, 0)
            ny1 = max(cy - size//2, 0)
            size = min(w - nx1, size)
            size = min(h - ny1, size)

            crop = frame[ny1:ny1+size, nx1:nx1+size]
            if crop.size > 0:
                img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                tensor = preprocess(pil_image.fromarray(img)).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = softmax(model(tensor))
                    fake_prob = output[0][1].item()  # probability of FAKE

                # Apply threshold
                if fake_prob >= fake_threshold:
                    label = 'FAKE'
                    conf = fake_prob
                    color = (0, 0, 255)
                    fake_count += 1
                elif fake_prob <= (1 - fake_threshold):
                    label = 'REAL'
                    conf = 1 - fake_prob
                    color = (0, 255, 0)
                    real_count += 1
                else:
                    label = 'UNCERTAIN'
                    conf = max(fake_prob, 1 - fake_prob)
                    color = (0, 165, 255)  # Orange
                    uncertain_count += 1

                fake_confidences.append(fake_prob)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f'{label} ({conf:.2f})', (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        writer.write(frame)

    cap.release()
    writer.release()
    pbar.close()

    total_f = real_count + fake_count + uncertain_count
    if total_f > 0:
        avg_fake_prob = np.mean(fake_confidences) if fake_confidences else 0
        fake_ratio = fake_count / total_f

        # Video-level verdict: require verdict_ratio% of frames to be FAKE
        if fake_ratio >= verdict_ratio:
            verdict = 'FAKE'
        else:
            verdict = 'REAL'

        print(f'\n🎯 VERDICT: {verdict}')
        print(f'   Average fake probability: {avg_fake_prob:.2%}')
        print(f'   Real frames:      {real_count} ({real_count/total_f*100:.1f}%)')
        print(f'   Fake frames:      {fake_count} ({fake_count/total_f*100:.1f}%)')
        print(f'   Uncertain frames: {uncertain_count} ({uncertain_count/total_f*100:.1f}%)')
        print(f'   Threshold: fake_prob >= {fake_threshold:.0%} | verdict requires {verdict_ratio:.0%} fake frames')
    print(f'   Output: {out_file}')


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--video_path', '-i', required=True)
    p.add_argument('--model_path', '-m', required=True)
    p.add_argument('--output_path', '-o', default='./predictions')
    p.add_argument('--threshold', '-t', type=float, default=0.70,
                   help='Minimum confidence to classify as FAKE (default: 0.70)')
    p.add_argument('--verdict_ratio', '-vr', type=float, default=0.6,
                   help='Fraction of frames needed as FAKE for video verdict (default: 0.60)')
    args = p.parse_args()
    predict_video(args.video_path, args.model_path, args.output_path,
                  args.threshold, args.verdict_ratio)

"""
PHASE 4: Run trained model on new video files.
Outputs annotated video + verdict.

Usage:
    python predict.py --video_path ./test_video.mp4 --model_path ./output/ff_effnet_v1/best.pth
"""
import os
import cv2
import torch
import torch.nn as nn
from PIL import Image as pil_image
from tqdm import tqdm
from facenet_pytorch import MTCNN

from network.models import model_selection
from dataset.transform import data_transforms


def predict_video(video_path, model_path, output_path='./predictions'):
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

    real_count = fake_count = 0
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
                    _, pred = torch.max(output, 1)

                label = 'FAKE' if pred.item() == 1 else 'REAL'
                conf = output[0][pred.item()].item()
                color = (0, 0, 255) if pred.item() == 1 else (0, 255, 0)

                if pred.item() == 1: fake_count += 1
                else: real_count += 1

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f'{label} ({conf:.2f})', (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        writer.write(frame)

    cap.release()
    writer.release()
    pbar.close()

    total_f = real_count + fake_count
    if total_f > 0:
        verdict = 'FAKE' if fake_count > real_count else 'REAL'
        print(f'\n🎯 VERDICT: {verdict}')
        print(f'   Real: {real_count} ({real_count/total_f*100:.1f}%)')
        print(f'   Fake: {fake_count} ({fake_count/total_f*100:.1f}%)')
    print(f'   Output: {out_file}')


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--video_path', '-i', required=True)
    p.add_argument('--model_path', '-m', required=True)
    p.add_argument('--output_path', '-o', default='./predictions')
    args = p.parse_args()
    predict_video(args.video_path, args.model_path, args.output_path)

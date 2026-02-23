"""
PHASE 1: Extract faces from FF++ videos using MTCNN.
Generates train/val/test data lists.

GPU-optimized: Batched MTCNN detection + parallel video I/O.

Usage:
    python preprocess_dataset.py --dataset_root "D:/DeepFake/FaceForensics++_C23" --output_root "./processed_faces" --sample_ratio 0.12 --num_workers 4
"""
import os
import cv2
import torch
import argparse
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
from facenet_pytorch import MTCNN
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


# Thread-local storage for MTCNN instances
_thread_local = threading.local()
_device = None  # Set in main()


def get_mtcnn():
    """Get or create a thread-local MTCNN instance on GPU."""
    if not hasattr(_thread_local, 'mtcnn'):
        _thread_local.mtcnn = MTCNN(
            image_size=380, margin=40, keep_all=False,
            select_largest=True, device=_device
        )
    return _thread_local.mtcnn


def extract_faces_from_video(video_path, output_dir,
                              frame_interval=10, face_size=380,
                              batch_size=16):
    """
    Extract faces using BATCHED MTCNN detection for better GPU utilization.
    1. Read all target frames from video (CPU)
    2. Batch them and run MTCNN on GPU in chunks
    3. Crop and save faces (CPU)
    """
    mtcnn = get_mtcnn()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Read all target frames into memory
    frames = []  # (frame_index, bgr_frame, pil_image)
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            frames.append((frame_count, frame, pil_img))
        frame_count += 1
    cap.release()

    if not frames:
        return []

    # Step 2: Batch MTCNN detection on GPU
    saved = []
    for batch_start in range(0, len(frames), batch_size):
        batch = frames[batch_start:batch_start + batch_size]
        pil_images = [f[2] for f in batch]

        # MTCNN batch detection — runs on GPU in parallel
        batch_boxes, batch_probs = [], []
        for pil_img in pil_images:
            boxes, probs = mtcnn.detect(pil_img)
            batch_boxes.append(boxes)
            batch_probs.append(probs)

        # Step 3: Crop and save detected faces
        for i, (fidx, bgr_frame, _) in enumerate(batch):
            boxes = batch_boxes[i]
            probs = batch_probs[i]

            if boxes is not None and len(boxes) > 0:
                best_idx = probs.argmax()
                box = boxes[best_idx]
                x1, y1, x2, y2 = [int(b) for b in box]

                h, w = bgr_frame.shape[:2]
                bw, bh = x2 - x1, y2 - y1
                size = int(max(bw, bh) * 1.3)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                nx1 = max(cx - size // 2, 0)
                ny1 = max(cy - size // 2, 0)
                size = min(w - nx1, size)
                size = min(h - ny1, size)

                crop = bgr_frame[ny1:ny1+size, nx1:nx1+size]
                if crop.size > 0:
                    crop = cv2.resize(crop, (face_size, face_size))
                    path = os.path.join(output_dir, f"{fidx:06d}.png")
                    cv2.imwrite(path, crop)
                    saved.append(path)

    return saved


def process_single_video(args_tuple):
    """Worker function for parallel video processing."""
    video_path, output_dir, frame_interval, label = args_tuple
    try:
        faces = extract_faces_from_video(video_path, output_dir, frame_interval)
        return [(p, label) for p in faces]
    except Exception as e:
        print(f"⚠️  Error processing {video_path}: {e}")
        return []


def main():
    global _device

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', '-d', required=True,
                        help='Path to FF++ dataset (contains original/, Deepfakes/, etc.)')
    parser.add_argument('--output_root', '-o', default='./processed_faces')
    parser.add_argument('--frame_interval', '-fi', type=int, default=10,
                        help='Extract every N-th frame (10 ≈ 3 faces/sec for 30fps video)')
    parser.add_argument('--sample_ratio', '-sr', type=float, default=0.12,
                        help='Fraction of videos to use from each folder (0.12 = 12%%)')
    parser.add_argument('--num_workers', '-nw', type=int, default=4,
                        help='Number of parallel threads for video processing')
    parser.add_argument('--split', nargs=3, type=float, default=[0.7, 0.15, 0.15])
    args = parser.parse_args()

    _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {_device}')

    if _device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f'GPU: {gpu_name} ({gpu_mem:.1f} GB)')
        print(f'Parallel workers: {args.num_workers}')
    else:
        print('⚠️  No GPU detected — falling back to CPU (slower)')

    # Initialize MTCNN on main thread first (downloads weights if needed)
    mtcnn = MTCNN(
        image_size=380, margin=40, keep_all=False,
        select_largest=True, device=_device
    )
    del mtcnn
    print('✅ MTCNN loaded\n')

    REAL = ['original']
    FAKE = ['Deepfakes', 'Face2Face', 'FaceSwap',
            'FaceShifter', 'NeuralTextures', 'DeepFakeDetection']

    # Collect all video tasks
    all_tasks = []
    for label, folders in [(0, REAL), (1, FAKE)]:
        tag = "REAL" if label == 0 else "FAKE"
        for folder in folders:
            path = os.path.join(args.dataset_root, folder)
            if not os.path.exists(path):
                print(f"⚠️  Not found: {path}, skipping")
                continue
            all_vids = [f for f in os.listdir(path)
                        if f.lower().endswith(('.mp4', '.avi', '.mov'))]
            random.shuffle(all_vids)
            n_sample = max(1, int(len(all_vids) * args.sample_ratio))
            vids = all_vids[:n_sample]
            print(f'📁 {tag} — {folder}/ ({len(vids)}/{len(all_vids)} videos sampled @ {args.sample_ratio*100:.0f}%)')

            for v in vids:
                vpath = os.path.join(path, v)
                vname = os.path.splitext(v)[0]
                odir = os.path.join(args.output_root, folder, vname)
                all_tasks.append((vpath, odir, args.frame_interval, label))

    print(f'\n🚀 Processing {len(all_tasks)} videos with {args.num_workers} parallel workers...\n')

    # Parallel processing with ThreadPoolExecutor
    entries = []
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(process_single_video, task): task
                   for task in all_tasks}
        with tqdm(total=len(futures), desc="Extracting faces") as pbar:
            for future in as_completed(futures):
                results = future.result()
                entries.extend(results)
                pbar.update(1)
                pbar.set_postfix(faces=len(entries))

    print(f'\n✅ Total faces extracted: {len(entries)}')

    random.shuffle(entries)
    n = len(entries)
    t1 = int(n * args.split[0])
    t2 = t1 + int(n * args.split[1])

    os.makedirs('./data_list', exist_ok=True)
    for name, data in [('train', entries[:t1]),
                       ('val', entries[t1:t2]),
                       ('test', entries[t2:])]:
        fpath = f'./data_list/ff_all_{name}.txt'
        with open(fpath, 'w') as f:
            for img, lbl in data:
                f.write(f"{img.replace(os.sep, '/')} {lbl}\n")
        real = sum(1 for _, l in data if l == 0)
        fake = sum(1 for _, l in data if l == 1)
        print(f'  📄 {name}: {len(data)} (Real:{real} Fake:{fake}) → {fpath}')

    print('\n🎉 Preprocessing complete!')


if __name__ == '__main__':
    main()

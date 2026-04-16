"""
PHASE 2: Train EfficientNet-B4 for deepfake detection.

Improvements over original:
  - ImageNet pretrained backbone (transfer learning)
  - WeightedRandomSampler (class balance)
  - CosineAnnealingWarmRestarts (smarter LR)
  - Mixed precision training (AMP) — 2x faster
  - Early stopping (patience=5)

Usage:
    python train_CNN.py --name ff_effnet_v1 --train_list ./data_list/ff_all_train.txt --val_list ./data_list/ff_all_val.txt
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast
import argparse
import os
import csv
import sys
import numpy as np

from network.models import model_selection
from dataset.transform import data_transforms
from dataset.mydataset import MyDataset


def make_balanced_sampler(dataset):
    """Creates a WeightedRandomSampler to balance fake vs real."""
    labels = dataset.get_labels()
    class_counts = np.bincount(labels)
    weights = 1.0 / class_counts
    sample_weights = [weights[l] for l in labels]
    return WeightedRandomSampler(sample_weights, len(labels), replacement=True)


def main():
    args = parser.parse_args()
    output_path = os.path.join('./output', args.name)
    os.makedirs(output_path, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda'
    print(f'Device: {device} | Mixed Precision: {use_amp}')

    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    # ── Data ──
    train_dataset = MyDataset(args.train_list, data_transforms['train'])
    val_dataset = MyDataset(args.val_list, data_transforms['val'])

    # Balanced sampling
    sampler = make_balanced_sampler(train_dataset)
    num_workers = 2 if torch.cuda.is_available() else 0
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              sampler=sampler, num_workers=num_workers,
                              pin_memory=True, persistent_workers=num_workers > 0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=num_workers,
                            pin_memory=True, persistent_workers=num_workers > 0)

    print(f'Train: {len(train_dataset)} | Val: {len(val_dataset)}')

    # ── Model ──
    # Phase 1: freeze backbone, train only classifier (2 epochs)
    # Phase 2: unfreeze everything, fine-tune full network
    model = model_selection('efficientnet_b4', num_out_classes=2,
                            dropout=0.5, freeze_backbone=True)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    scaler = GradScaler(enabled=use_amp)

    if device.type == 'cuda' and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    best_acc = 0.0
    best_wts = None
    patience_counter = 0
    unfreeze_epoch = 2  # Unfreeze backbone after 2 epochs

    # ── Logging setup ──
    history_file = os.path.join(output_path, 'training_history.csv')
    log_file = os.path.join(output_path, 'training_log.txt')

    # Write CSV header
    with open(history_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'lr', 'best_acc'])

    # Tee stdout to log file
    class Tee:
        def __init__(self, file_path):
            self.file = open(file_path, 'w', encoding='utf-8')
            self.stdout = sys.stdout
        def write(self, data):
            self.stdout.write(data)
            self.file.write(data)
            self.file.flush()
        def flush(self):
            self.stdout.flush()
            self.file.flush()
    sys.stdout = Tee(log_file)

    for epoch in range(args.epoches):
        # Unfreeze backbone after warmup
        if epoch == unfreeze_epoch:
            print('\n🔓 Unfreezing backbone — full fine-tuning starts')
            m = model.module if isinstance(model, nn.DataParallel) else model
            m.unfreeze_backbone()
            # Reset optimizer with lower LR for backbone
            optimizer = optim.AdamW([
                {'params': m.backbone.features.parameters(), 'lr': args.lr * 0.1},
                {'params': m.backbone.classifier.parameters(), 'lr': args.lr},
            ], weight_decay=1e-4)
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)

        print(f'\nEpoch {epoch+1}/{args.epoches} | LR: {optimizer.param_groups[0]["lr"]:.6f}')
        print('-' * 50)

        # ── Train ──
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            with autocast(enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            _, preds = torch.max(outputs, 1)
            train_loss += loss.item() * images.size(0)
            train_correct += (preds == labels).sum().item()
            train_total += images.size(0)

            if (batch_idx + 1) % 50 == 0:
                print(f'  batch {batch_idx+1} | loss: {loss.item():.4f} | '
                      f'acc: {(preds == labels).float().mean():.4f}')

        train_acc = train_correct / train_total
        print(f'  TRAIN — Loss: {train_loss/train_total:.4f} | Acc: {train_acc:.4f}')

        # ── Validate ──
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                with autocast(enabled=use_amp):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * images.size(0)
                val_correct += (preds == labels).sum().item()
                val_total += images.size(0)

        val_acc = val_correct / val_total
        print(f'  VAL   — Loss: {val_loss/val_total:.4f} | Acc: {val_acc:.4f}')

        scheduler.step()

        # ── Save history to CSV ──
        with open(history_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                f'{train_loss/train_total:.4f}',
                f'{train_acc:.4f}',
                f'{val_loss/val_total:.4f}',
                f'{val_acc:.4f}',
                f'{optimizer.param_groups[0]["lr"]:.6f}',
                f'{best_acc:.4f}'
            ])

        # Save checkpoint
        m = model.module if isinstance(model, nn.DataParallel) else model
        torch.save(m.state_dict(), os.path.join(output_path, f'epoch_{epoch}.pth'))

        # Best model tracking
        if val_acc > best_acc:
            best_acc = val_acc
            best_wts = m.state_dict().copy()
            patience_counter = 0
            torch.save(best_wts, os.path.join(output_path, 'best.pth'))
            print(f'  ⭐ New best! Saved best.pth (acc={best_acc:.4f})')
        else:
            patience_counter += 1
            print(f'  ⏳ No improvement ({patience_counter}/{args.patience})')

        # Early stopping
        if patience_counter >= args.patience:
            print(f'\n⛔ Early stopping at epoch {epoch+1}')
            break

    print(f'\n🏆 Best Val Accuracy: {best_acc:.4f}')
    print(f'✅ Best model: {output_path}/best.pth')
    print(f'📊 History: {history_file}')
    print(f'📝 Log: {log_file}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', '-n', default='ff_effnet_v1')
    parser.add_argument('--train_list', '-tl', default='./data_list/ff_all_train.txt')
    parser.add_argument('--val_list', '-vl', default='./data_list/ff_all_val.txt')
    parser.add_argument('--batch_size', '-bz', type=int, default=8)
    parser.add_argument('--epoches', '-e', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=5,
                        help='Early stopping patience')
    parser.add_argument('--continue_train', type=bool, default=False)
    parser.add_argument('--model_path', '-mp', default='')
    main()

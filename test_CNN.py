"""
PHASE 3: Evaluate trained model on test set.
Reports: Accuracy, Precision, Recall, F1, Confusion Matrix.

Usage:
    python test_CNN.py --test_list ./data_list/ff_all_test.txt --model_path ./output/ff_effnet_v1/best.pth
"""
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import argparse

from network.models import model_selection
from dataset.transform import data_transforms
from dataset.mydataset import MyDataset


def main():
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_dataset = MyDataset(args.test_list, data_transforms['test'])
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = model_selection('efficientnet_b4', num_out_classes=2, dropout=0.5)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    model.eval()

    tp = fp = tn = fn = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            with autocast(enabled=device.type == 'cuda'):
                outputs = model(images)
            _, preds = torch.max(outputs, 1)

            tp += ((preds == 1) & (labels == 1)).sum().item()
            fp += ((preds == 1) & (labels == 0)).sum().item()
            tn += ((preds == 0) & (labels == 0)).sum().item()
            fn += ((preds == 0) & (labels == 1)).sum().item()

    total = tp + fp + tn + fn
    acc = (tp + tn) / total if total else 0
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0

    print(f'\n📊 Test Results ({total} images)')
    print(f'   Accuracy:  {acc:.4f}  ({acc*100:.1f}%)')
    print(f'   Precision: {prec:.4f}')
    print(f'   Recall:    {rec:.4f}')
    print(f'   F1 Score:  {f1:.4f}')
    print(f'\n   Confusion Matrix:')
    print(f'                 Predicted')
    print(f'              REAL    FAKE')
    print(f'   Real  |  {tn:>6}  {fp:>6}')
    print(f'   Fake  |  {fn:>6}  {tp:>6}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', '-bz', type=int, default=32)
    parser.add_argument('--test_list', '-tl', default='./data_list/ff_all_test.txt')
    parser.add_argument('--model_path', '-mp', required=True)
    main()

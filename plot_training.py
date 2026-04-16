"""
📊 Plot Training Curves — Loss and Accuracy per epoch.
Reads training_history.csv and generates publication-quality charts.

Usage:
    python plot_training.py --history ./output/ff_effnet_v2/training_history.csv
"""
import os
import csv
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def load_history(csv_path):
    """Load training history from CSV."""
    history = {'epoch': [], 'train_loss': [], 'train_acc': [],
               'val_loss': [], 'val_acc': [], 'lr': [], 'best_acc': []}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            history['epoch'].append(int(row['epoch']))
            history['train_loss'].append(float(row['train_loss']))
            history['train_acc'].append(float(row['train_acc']))
            history['val_loss'].append(float(row['val_loss']))
            history['val_acc'].append(float(row['val_acc']))
            history['lr'].append(float(row['lr']))
            history['best_acc'].append(float(row['best_acc']))
    return history


def plot_training_curves(history, output_dir):
    """Generate and save all training charts."""
    os.makedirs(output_dir, exist_ok=True)
    epochs = history['epoch']

    # ── Style ──
    plt.rcParams.update({
        'figure.facecolor': '#0f172a',
        'axes.facecolor': '#0f172a',
        'axes.edgecolor': '#334155',
        'text.color': '#e2e8f0',
        'axes.labelcolor': '#e2e8f0',
        'xtick.color': '#94a3b8',
        'ytick.color': '#94a3b8',
        'grid.color': '#1e293b',
        'font.size': 12,
    })

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  1. Loss vs Epochs
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, history['train_loss'], 'o-', color='#667eea',
            linewidth=2.5, markersize=8, label='Train Loss')
    ax.plot(epochs, history['val_loss'], 's-', color='#f472b6',
            linewidth=2.5, markersize=8, label='Val Loss')
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Loss', fontsize=14)
    ax.set_title('Training & Validation Loss', fontsize=16, fontweight='bold', pad=15)
    ax.legend(fontsize=12, facecolor='#1e293b', edgecolor='#334155')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(epochs)

    # Mark unfreeze point
    if len(epochs) > 2:
        ax.axvline(x=3, color='#f59e0b', linestyle='--', alpha=0.7, label='Backbone Unfrozen')
        ax.text(3.1, max(history['train_loss']) * 0.9, '🔓 Unfreeze',
                color='#f59e0b', fontsize=10)

    fig.tight_layout()
    loss_path = os.path.join(output_dir, 'loss_curve.png')
    fig.savefig(loss_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  📈 Loss curve → {loss_path}')

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  2. Accuracy vs Epochs
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, [a * 100 for a in history['train_acc']], 'o-', color='#34d399',
            linewidth=2.5, markersize=8, label='Train Accuracy')
    ax.plot(epochs, [a * 100 for a in history['val_acc']], 's-', color='#f59e0b',
            linewidth=2.5, markersize=8, label='Val Accuracy')
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Accuracy (%)', fontsize=14)
    ax.set_title('Training & Validation Accuracy', fontsize=16, fontweight='bold', pad=15)
    ax.legend(fontsize=12, facecolor='#1e293b', edgecolor='#334155')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(epochs)
    ax.set_ylim(0, 105)

    # Mark best accuracy
    best_epoch = epochs[np.argmax(history['val_acc'])]
    best_val = max(history['val_acc']) * 100
    ax.annotate(f'Best: {best_val:.1f}%', xy=(best_epoch, best_val),
                xytext=(best_epoch + 0.5, best_val - 8),
                arrowprops=dict(arrowstyle='->', color='#f59e0b', lw=2),
                fontsize=12, fontweight='bold', color='#f59e0b')

    if len(epochs) > 2:
        ax.axvline(x=3, color='#f59e0b', linestyle='--', alpha=0.7)
        ax.text(3.1, 15, '🔓 Unfreeze', color='#f59e0b', fontsize=10)

    fig.tight_layout()
    acc_path = os.path.join(output_dir, 'accuracy_curve.png')
    fig.savefig(acc_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  📈 Accuracy curve → {acc_path}')

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  3. Combined (Loss + Accuracy)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Loss
    ax1.plot(epochs, history['train_loss'], 'o-', color='#667eea', linewidth=2.5, markersize=7, label='Train')
    ax1.plot(epochs, history['val_loss'], 's-', color='#f472b6', linewidth=2.5, markersize=7, label='Val')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss', fontsize=14, fontweight='bold')
    ax1.legend(facecolor='#1e293b', edgecolor='#334155')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(epochs)

    # Accuracy
    ax2.plot(epochs, [a * 100 for a in history['train_acc']], 'o-', color='#34d399', linewidth=2.5, markersize=7, label='Train')
    ax2.plot(epochs, [a * 100 for a in history['val_acc']], 's-', color='#f59e0b', linewidth=2.5, markersize=7, label='Val')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(facecolor='#1e293b', edgecolor='#334155')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(epochs)
    ax2.set_ylim(0, 105)

    fig.suptitle('DeepFake Detection — Training Summary', fontsize=16, fontweight='bold', y=1.02)
    fig.tight_layout()
    combined_path = os.path.join(output_dir, 'training_summary.png')
    fig.savefig(combined_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  📈 Combined chart → {combined_path}')

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  4. Results Table (text)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    table_path = os.path.join(output_dir, 'training_results_table.txt')
    with open(table_path, 'w') as f:
        f.write(f"{'Epoch':>6} | {'Train Loss':>11} | {'Train Acc':>10} | {'Val Loss':>9} | {'Val Acc':>8} | {'LR':>10} | {'Best':>6}\n")
        f.write("-" * 80 + "\n")
        for i, ep in enumerate(epochs):
            f.write(f"{ep:>6} | {history['train_loss'][i]:>11.4f} | {history['train_acc'][i]*100:>9.2f}% | "
                    f"{history['val_loss'][i]:>9.4f} | {history['val_acc'][i]*100:>7.2f}% | "
                    f"{history['lr'][i]:>10.6f} | {history['best_acc'][i]*100:>5.2f}%\n")
        f.write("-" * 80 + "\n")
        f.write(f"\n🏆 Best Validation Accuracy: {max(history['val_acc'])*100:.2f}% (Epoch {epochs[np.argmax(history['val_acc'])]})\n")
    print(f'  📋 Results table → {table_path}')

    # Print table to console too
    with open(table_path, 'r') as f:
        print(f.read())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--history', '-H', default='./output/ff_effnet_v2/training_history.csv')
    parser.add_argument('--output', '-o', default=None,
                        help='Output directory (default: same as history file)')
    args = parser.parse_args()

    if not os.path.exists(args.history):
        print(f"❌ History file not found: {args.history}")
        return

    output_dir = args.output or os.path.dirname(args.history)
    history = load_history(args.history)

    print(f"\n📊 Plotting training curves ({len(history['epoch'])} epochs)...\n")
    plot_training_curves(history, output_dir)
    print(f"\n🎉 All charts saved to: {output_dir}")


if __name__ == '__main__':
    main()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
from sklearn.metrics import confusion_matrix

def plot_training_curves(log_file, output_dir='results'):
    if not os.path.exists(log_file):
        print(f"Warning: {log_file} not found. Skipping training curves.")
        return

    df = pd.read_csv(log_file)
    model_name = os.path.basename(log_file).replace('training_log_', '').replace('.csv', '')

    # Plot Loss
    plt.figure(figsize=(10, 5))
    plt.plot(df['epoch'], df['train_loss'], label='Train Loss', marker='o')
    plt.title(f'Training Loss Curve ({model_name})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'loss_curve_{model_name}.png'))
    plt.close()

    # Plot Accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(df['epoch'], df['train_acc'], label='Train Acc', marker='o')
    plt.plot(df['epoch'], df['val_acc'], label='Val Acc', marker='x')
    plt.title(f'Accuracy Curve ({model_name})')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'accuracy_curve_{model_name}.png'))
    plt.close()
    
    print(f"Generated training curves for {model_name} in {output_dir}/")

def plot_confusion_matrix(y_true, y_pred_probs, model_name, split_name, output_dir='results'):
    y_pred = (np.array(y_pred_probs) > 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-Hateful', 'Hateful'], 
                yticklabels=['Non-Hateful', 'Hateful'])
    plt.title(f'Confusion Matrix - {model_name} ({split_name})')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(os.path.join(output_dir, f'confusion_matrix_{model_name}_{split_name}.png'))
    plt.close()
    print(f"Generated confusion matrix for {model_name} in {output_dir}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_file', type=str, help='Path to training_log.csv')
    args = parser.parse_args()

    os.makedirs('results', exist_ok=True)
    
    if args.log_file:
        plot_training_curves(args.log_file)
    else:
        # Auto-detect logs in results folder
        for f in os.listdir('results'):
            if f.startswith('training_log_') and f.endswith('.csv'):
                plot_training_curves(os.path.join('results', f))

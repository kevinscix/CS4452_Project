import os
import random
import argparse
import csv
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np
from src.data_loader import get_data_loaders
from src.models import MultimodalModel, UnimodalTextModel, UnimodalImageModel
from src.utils import calculate_metrics

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train(model, train_loader, val_loader, device, epochs=5, lr=2e-5, model_type='multimodal'):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=lr)

    total_steps = len(train_loader) * epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    model.to(device)

    best_val_auc = 0.0
    training_history = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        all_labels = []
        all_preds = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['label'].to(device).unsqueeze(1)

            optimizer.zero_grad()

            if model_type == 'text':
                logits = model(input_ids, attention_mask)
            elif model_type == 'image':
                logits = model(pixel_values=pixel_values)
            else:
                logits = model(input_ids, attention_mask, pixel_values)

            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(probs)

            pbar.set_postfix({'loss': train_loss / (pbar.n + 1)})

        train_acc, train_f1, train_auc = calculate_metrics(np.array(all_labels).flatten(), np.array(all_preds).flatten())
        print(f"Train - Acc: {train_acc:.4f}, F1: {train_f1:.4f}, AUC: {train_auc:.4f}, Loss: {train_loss/len(train_loader):.4f}")

        # Validation
        model.eval()
        val_labels = []
        val_preds = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                pixel_values = batch['pixel_values'].to(device)
                labels = batch['label'].to(device).unsqueeze(1)

                if model_type == 'text':
                    logits = model(input_ids, attention_mask)
                elif model_type == 'image':
                    logits = model(pixel_values=pixel_values)
                else:
                    logits = model(input_ids, attention_mask, pixel_values)

                probs = torch.sigmoid(logits).cpu().numpy()
                val_labels.extend(labels.cpu().numpy())
                val_preds.extend(probs)

        val_acc, val_f1, val_auc = calculate_metrics(np.array(val_labels).flatten(), np.array(val_preds).flatten())
        print(f"Val - Acc: {val_acc:.4f}, F1: {val_f1:.4f}, AUC: {val_auc:.4f}")

        # Save checkpoint and track best model by val AUC
        torch.save(model.state_dict(), f"model_{model_type}_{epoch}.pt")
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), f"model_{model_type}_best.pt")
            print(f"  -> New best model saved (val AUC: {best_val_auc:.4f})")

        # Save metrics to history
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss / len(train_loader),
            'train_acc': train_acc,
            'train_f1': train_f1,
            'train_auc': train_auc,
            'val_acc': val_acc,
            'val_f1': val_f1,
            'val_auc': val_auc
        })

    # Save training history to CSV
    os.makedirs('results', exist_ok=True)
    csv_file = os.path.join('results', f'training_log_{model_type}.csv')
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=training_history[0].keys())
        writer.writeheader()
        writer.writerows(training_history)
    print(f"\nTraining history saved to {csv_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='multimodal', choices=['multimodal', 'text', 'image'])
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--img_dir', type=str, default='.')
    args = parser.parse_args()
    
    set_seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    train_loader, dev_loader, test_loader = get_data_loaders(args.data_dir, img_dir=args.img_dir, batch_size=args.batch_size)
    
    if args.model_type == 'multimodal':
        model = MultimodalModel()
    elif args.model_type == 'text':
        model = UnimodalTextModel()
    elif args.model_type == 'image':
        model = UnimodalImageModel()
    
    train(model, train_loader, dev_loader, device, epochs=args.epochs, lr=args.lr, model_type=args.model_type)

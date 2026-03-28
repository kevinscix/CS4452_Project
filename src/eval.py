import os
import argparse
import csv
import torch
from tqdm import tqdm
import numpy as np
import json
from src.data_loader import get_data_loaders
from src.models import MultimodalModel, UnimodalTextModel, UnimodalImageModel
from src.utils import calculate_metrics
from src.plots import plot_confusion_matrix

def evaluate(model, loader, device, model_type='multimodal', split_name='Dev'):
    model.eval()
    all_labels = []
    all_preds = []
    all_indices = []

    idx_start = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Evaluating {split_name}"):
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
            actual_batch_size = probs.shape[0]

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(probs)
            all_indices.extend(range(idx_start, idx_start + actual_batch_size))
            idx_start += actual_batch_size

    labels_flat = np.array(all_labels).flatten()
    preds_flat = np.array(all_preds).flatten()

    acc, f1, auc = calculate_metrics(labels_flat, preds_flat)
    print(f"{split_name} - Acc: {acc:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")

    return labels_flat, preds_flat, all_indices

def qualitative_analysis(jsonl_file, labels, preds, indices, num_examples=5):
    data = []
    with open(jsonl_file, 'r', encoding='utf-8-sig') as f:
        for line in f:
            data.append(json.loads(line))

    false_positives = []
    false_negatives = []

    for i in range(len(labels)):
        true_label = int(labels[i])
        pred_prob = float(preds[i])
        pred_label = 1 if pred_prob > 0.5 else 0

        if true_label == pred_label:
            continue

        entry = {
            'id': data[indices[i]]['id'],
            'text': data[indices[i]]['text'],
            'img': data[indices[i]]['img'],
            'true_label': true_label,
            'pred_prob': pred_prob,
            'confidence': abs(pred_prob - 0.5),  # distance from decision boundary
        }

        if pred_label == 1 and true_label == 0:
            false_positives.append(entry)
        else:
            false_negatives.append(entry)

    # Sort each group by confidence (most wrong = furthest from boundary)
    false_positives.sort(key=lambda x: x['confidence'], reverse=True)
    false_negatives.sort(key=lambda x: x['confidence'], reverse=True)

    top_fp = false_positives[:num_examples]
    top_fn = false_negatives[:num_examples]

    print(f"\nFound {len(false_positives) + len(false_negatives)} misclassified examples.")
    return top_fp, top_fn

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='multimodal', choices=['multimodal', 'text', 'image'])
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--img_dir', type=str, default='.')
    parser.add_argument('--split', type=str, default='dev', choices=['dev', 'test'])
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    _, dev_loader, test_loader = get_data_loaders(args.data_dir, img_dir=args.img_dir, batch_size=16)
    
    loader = dev_loader if args.split == 'dev' else test_loader
    jsonl_filename = 'dev.jsonl' if args.split == 'dev' else 'test.jsonl'

    if args.model_type == 'multimodal':
        model = MultimodalModel()
    elif args.model_type == 'text':
        model = UnimodalTextModel()
    elif args.model_type == 'image':
        model = UnimodalImageModel()

    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)

    print(f"\nEvaluating {args.model_type} model on {args.split.capitalize()} set:")
    labels, preds, indices = evaluate(model, loader, device, args.model_type, split_name=args.split.capitalize())
    
    # Save metrics to summary CSV
    os.makedirs('results', exist_ok=True)
    summary_file = os.path.join('results', f'summary_{args.model_type}_{args.split}.csv')
    acc, f1, auc = calculate_metrics(labels, preds)
    with open(summary_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['model_type', 'split', 'acc', 'f1', 'auc'])
        writer.writeheader()
        writer.writerow({'model_type': args.model_type, 'split': args.split, 'acc': acc, 'f1': f1, 'auc': auc})
    
    # Generate Confusion Matrix Figure
    plot_confusion_matrix(labels, preds, args.model_type, args.split, output_dir='results')
    
    # Get and save top mistakes
    top_fp, top_fn = qualitative_analysis(os.path.join(args.data_dir, jsonl_filename), labels, preds, indices)
    
    mistakes_file = os.path.join('results', f'mistakes_{args.model_type}_{args.split}.csv')
    with open(mistakes_file, 'w', newline='', encoding='utf-8') as f:
        if top_fp or top_fn:
            writer = csv.DictWriter(f, fieldnames=['category', 'id', 'true_label', 'pred_prob', 'confidence', 'text', 'img'])
            writer.writeheader()
            for ex in top_fp:
                ex['category'] = 'False Positive (Predicted Hateful)'
                writer.writerow(ex)
            for ex in top_fn:
                ex['category'] = 'False Negative (Predicted Non-Hateful)'
                writer.writerow(ex)
    
    print(f"\nResults saved to results/ folder:")
    print(f"  - Metrics: {summary_file}")
    print(f"  - Top Mistakes: {mistakes_file}")

import json
import random
from sklearn.model_selection import train_test_split
import os

def split_dataset(input_file, output_dir, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2, seed=42):
    # Load all data, only keeping labeled items
    labeled_data = []
    
    with open(input_file, 'r', encoding='utf-8-sig') as f:
        for line in f:
            item = json.loads(line)
            if 'label' in item:
                labeled_data.append(item)
    
    # Extract labels for stratification
    labels = [item['label'] for item in labeled_data]
    
    # First split: Train vs (Val + Test)
    rest_ratio = val_ratio + test_ratio
    train_data, rest_data, train_labels, rest_labels = train_test_split(
        labeled_data, labels, test_size=rest_ratio, stratify=labels, random_state=seed
    )
    
    # Second split: Val vs Test
    relative_val_ratio = val_ratio / rest_ratio
    val_data, test_data = train_test_split(
        rest_data, test_size=(1 - relative_val_ratio), stratify=rest_labels, random_state=seed
    )
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Helper to save jsonl
    def save_jsonl(data, filename):
        with open(os.path.join(output_dir, filename), 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
    
    # Save the splits
    save_jsonl(train_data, 'train.jsonl')
    save_jsonl(val_data, 'dev.jsonl')
    save_jsonl(test_data, 'test.jsonl')
    
    print(f"Split complete (unlabeled data dropped):")
    print(f"  Train: {len(train_data)} ({len(train_data)/len(labeled_data):.1%})")
    print(f"  Val:   {len(val_data)} ({len(val_data)/len(labeled_data):.1%})")
    print(f"  Test:  {len(test_data)} ({len(test_data)/len(labeled_data):.1%})")
    print(f"  Total Labeled: {len(labeled_data)}")

if __name__ == "__main__":
    split_dataset('master_list.jsonl', 'data', train_ratio=0.7, val_ratio=0.1, test_ratio=0.2)

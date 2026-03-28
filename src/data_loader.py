import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import RobertaTokenizer, ViTImageProcessor

class HatefulMemesDataset(Dataset):
    def __init__(self, jsonl_file, img_dir, tokenizer, image_processor, max_length=128):
        self.data = []
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line))
        
        self.img_dir = img_dir
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        img_path = os.path.join(self.img_dir, item['img'])
        label = item.get('label', 0)  # test set might not have labels

        # Process text
        text_inputs = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        # Process image
        image = Image.open(img_path).convert('RGB')
        image_inputs = self.image_processor(image, return_tensors='pt')

        return {
            'input_ids': text_inputs['input_ids'].squeeze(0),
            'attention_mask': text_inputs['attention_mask'].squeeze(0),
            'pixel_values': image_inputs['pixel_values'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.float)
        }

def get_data_loaders(data_dir, img_dir='.', batch_size=16, max_length=128):
    tokenizer = RobertaTokenizer.from_pretrained('FacebookAI/roberta-base')
    image_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')

    train_dataset = HatefulMemesDataset(
        os.path.join(data_dir, 'train.jsonl'),
        img_dir,
        tokenizer,
        image_processor,
        max_length
    )
    dev_dataset = HatefulMemesDataset(
        os.path.join(data_dir, 'dev.jsonl'),
        img_dir,
        tokenizer,
        image_processor,
        max_length
    )
    test_dataset = HatefulMemesDataset(
        os.path.join(data_dir, 'test.jsonl'),
        img_dir,
        tokenizer,
        image_processor,
        max_length
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, dev_loader, test_loader

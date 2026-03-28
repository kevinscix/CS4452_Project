import torch
import torch.nn as nn
from transformers import RobertaModel, ViTModel

class TextEncoder(nn.Module):
    def __init__(self, model_name='FacebookAI/roberta-base'):
        super(TextEncoder, self).__init__()
        self.roberta = RobertaModel.from_pretrained(model_name)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        # Use the [CLS] token representation
        return outputs.last_hidden_state[:, 0, :]

class ImageEncoder(nn.Module):
    def __init__(self, model_name='google/vit-base-patch16-224'):
        super(ImageEncoder, self).__init__()
        self.vit = ViTModel.from_pretrained(model_name)
    
    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        # Use the [CLS] token representation
        return outputs.last_hidden_state[:, 0, :]

class MultimodalModel(nn.Module):
    def __init__(self, text_hidden_size=768, image_hidden_size=768, num_labels=1):
        super(MultimodalModel, self).__init__()
        self.text_encoder = TextEncoder()
        self.image_encoder = ImageEncoder()
        
        self.classifier = nn.Sequential(
            nn.Linear(text_hidden_size + image_hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_labels)
        )
    
    def forward(self, input_ids, attention_mask, pixel_values):
        text_features = self.text_encoder(input_ids, attention_mask)
        image_features = self.image_encoder(pixel_values)
        
        combined_features = torch.cat((text_features, image_features), dim=1)
        logits = self.classifier(combined_features)
        return logits

class UnimodalTextModel(nn.Module):
    def __init__(self, text_hidden_size=768, num_labels=1):
        super(UnimodalTextModel, self).__init__()
        self.text_encoder = TextEncoder()
        self.classifier = nn.Sequential(
            nn.Linear(text_hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_labels)
        )
    
    def forward(self, input_ids, attention_mask, pixel_values=None):
        text_features = self.text_encoder(input_ids, attention_mask)
        logits = self.classifier(text_features)
        return logits

class UnimodalImageModel(nn.Module):
    def __init__(self, image_hidden_size=768, num_labels=1):
        super(UnimodalImageModel, self).__init__()
        self.image_encoder = ImageEncoder()
        self.classifier = nn.Sequential(
            nn.Linear(image_hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_labels)
        )
    
    def forward(self, input_ids=None, attention_mask=None, pixel_values=None):
        image_features = self.image_encoder(pixel_values)
        logits = self.classifier(image_features)
        return logits

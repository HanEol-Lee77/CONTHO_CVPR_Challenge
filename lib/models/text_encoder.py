import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoTokenizer, CLIPTextModel

class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        self.tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.out_dim = self.model.text_model.final_layer_norm.normalized_shape[0]

    def forward(self, x: str) -> torch.Tensor:
        x = self.tokenizer(
            x,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids

        x = self.model(x)[0]
        
        return x
    
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from config import CFG

class Model(nn.Module):
    def __init__(self, base_model_ckpt) -> None:
        super().__init__()
        self.model = AutoModel.from_pretrained(base_model_ckpt)
        self.linear = nn.Linear(768, 1)     # 768: Size of output bert encode, 1: Binary classification
        self.dropout = nn.Dropout(p=CFG.hidden_prob_drop)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids, attention_mask, output_hidden_states=True)
        cls_token = outputs.last_hidden_state[:,0, ...]
        outputs = self.dropout(cls_token)
        outputs = self.linear(outputs)
        outputs = self.sigmoid(outputs)
        return outputs

def create_model(base_model_ckpt):
    return Model(base_model_ckpt)

def create_tokenizer(base_model_ckpt):
    return AutoTokenizer.from_pretrained(base_model_ckpt)
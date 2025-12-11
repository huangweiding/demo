import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from typing import List
from transformers import PreTrainedTokenizer, AutoTokenizer
import json
from models.miniModel import miniModelForCausalLM, MiniConfig
from torch.utils.data import Dataset
import torch

def init_model(config: MiniConfig, tokenizer_path: str, weight_path: str=None, device: str="cuda"):
    model = miniModelForCausalLM(config)
    if weight_path:
        weights = torch.load(weight_path, map_location=device)
        model.load_state_dict(weights)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    return model.to(device), tokenizer


class PretrainDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: PreTrainedTokenizer, max_length: int=512):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samplers = self.load_data(self.data_path)

    def __len__(self):
        return len(self.samplers)
    
    def load_data(self, data_path: str):
        # assume data_path is a jsonl file
        with open(data_path) as f:
            return [line for line in json.load(f)]

    def __getitem__(self, idx):
        sample = self.samplers[idx]
        encoding = self.tokenizer(sample['text'], padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
        input_ids = encoding.input_ids.squeeze()
        mask = input_ids != self.tokenizer.pad_token_id
        X = input_ids[:-1]
        y = input_ids[1:]
        mask = mask[1:]
        return X, y, mask
    

if __name__ == "__main__":
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    dataset = PretrainDataset(data_path="/repos/tmp/demo/MiniModel/data/test_data.jsonl", tokenizer=tokenizer)
    x, y, mask = dataset[0]
    print(x)
    print('='*100)
    print(y)
    print('='*100)
    print(mask)

    
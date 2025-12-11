from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
import json

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
        return X, y, mask
    

if __name__ == "__main__":
    from transformers import AutoTokenizer
    from torch.utils.data.distributed import DistributedSampler
    from torch.utils.data import DataLoader
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    dataset = PretrainDataset(data_path="/repos/tmp/demo/MiniModel/data/test_data.jsonl", tokenizer=tokenizer)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    for X, y, mask in dataloader:
        print(X)
        print('='*100)
        print(y)
        print('='*100)
        print(mask)

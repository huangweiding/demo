import torch
class MyModel(torch.nn.Module):
    def __init__(self,hidden_size, vocab_size, num_heads, intermediate_size, ):
        super().__init__()

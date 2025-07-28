import torch

class SimpleExpert(torch.nn.Module):
    def __init__(self, hidden_size, intermediate_size, bias=False):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate = torch.nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up = torch.nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down = torch.nn.Linear(intermediate_size, hidden_size, bias=bias)

        self.act_fn = torch.nn.functional.silu

    def forward(self, x):
        return self.down(self.act_fn(self.gate(x))* self.up(x))




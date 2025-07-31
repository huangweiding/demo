import torch
class RMSNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        self.weights = torch.nn.Parameter(torch.ones((hidden_size, hidden_size)))
        self.eps = eps

    def forward(self, inputs, attention_mask=None):
        rms_norm = torch.rsqrt(inputs.pow(2).mean(-1, keepdim=True)+self.eps)
        return inputs * rms_norm* self.weights



class AttentionBlock(torch.nn.Module):
    def __init__(self, hidden_size, intermedaite_size, num_heads, sliding_window):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.intermediate_size = intermedaite_size
        self.num_heads = num_heads
        self.sliding_window = sliding_window
        self.head_dim = hidden_size // num_heads

        self.q_proj = torch.nn.Linear(hidden_size, hidden_size)
        self.k_proj = torch.nn.Linear(hidden_size, hidden_size)
        self.v_proj = torch.nn.Linear(hidden_size, hidden_size)
        self.o_proj = torch.nn.Linear(hidden_size, hidden_size)

        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)


    def forward(self, inputs, attention_mask=None):
        batch_size, seq_length, hidden_size = inputs.shape

        Q = self.q_proj(inputs).view(batch_size, seq_length, self.num_heads, -1).transpose(-2, -1)
        K = self.k_proj(inputs).view(batch_size, seq_length, self.num_heads, -1).transpose(-2, -1)
        V = self.v_proj(inputs).view(batch_size, seq_length, self.num_heads, -1).transpose(-2, -1)

        Q = self.q_norm(Q)
        K = self.k_norm(K)


def create_attention_mask():
    pass


class MyModel(torch.nn.Module):
    def __init__(self,hidden_size, vocab_size, num_heads, intermediate_size, sliding_window, num_layers, num_max_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.sliding_window = sliding_window
        self.num_layers = num_layers
        self.num_max_layers = num_max_layers

        self.embedding_layer = torch.nn.Embedding(vocab_size, hidden_size)

        self.attention_layers = torch.nn.ModuleList()
        self.layer_type = []

        for i in range(num_layers):
            if i < num_max_layers:
                self.layer_type.append("full_attention")
            else:
                self.layer_type.append("sliding_window_attention")

        for i in range(num_layers):
            if self.layer_type[i] == "full_attention":
                current_layer = AttentionBlock()
            else:
                current_layer = AttentionBlock()
            self.attention_layers.append(current_layer)
        

    def forward(self, attention_mask=None):
        pass


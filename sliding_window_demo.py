import torch
class RMSNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.weights = torch.nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        # [BS, seq_length, hidden_size]
        rms_norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True)+self.eps)
        return x*rms_norm*self.weights

def apply_sliding_attention_mask(hidden_size, num_heads, sliding_window, mask=None):
    


class SlidingWindowLayer(torch.nn.Module):
    def __init__(self, hidden_size, num_heads, sliding_window, mask=None):
        assert hidden_size % num_heads != 0
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.sliding_window = sliding_window
        self.mask = mask

        self.q_proj = torch.Linear(hidden_size, hidden_size)
        self.k_proj = torch.Linear(hidden_size, hidden_size)
        self.v_proj = torch.Linear(hidden_size, hidden_size)
        self.o_proj = torch.Linear(hidden_size, hidden_size)

        self.q_norm = RMSNorm(hidden_size)
        self.k_norm = RMSNorm(hidden_size)




    def forward(self, inputs, attention_mask=None):
        
        batch_size, seq_length, hidden_size = inputs.shape

        Q = self.q_proj(inputs).view(batch_size, seq_length, self.num_heads, self.hidden_size//self.num_heads).transpose(1, 2)
        K = self.k_proj(inputs).view(batch_size, seq_length, self.num_heads, self.hidden_size//self.num_heads).transpose(1, 2)
        V = self.v_proj(inputs).view(batch_size, seq_length, self.num_heads, self.hidden_size//self.num_heads).transpose(1, 2)




        
        return inputs

class MyModel(torch.nn.Module):
    def __init__(self, vocab_size, hidden_size, num_heads, num_layers, sliding_window, max_num_layer):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.sliding_window = sliding_window
        self.max_num_layer = max_num_layer

        self.embedding_layer = torch.nn.Embedding(vocab_size, hidden_size)

        self.layers = torch.nn.ModuleList()

        self.layer_types = []
        for i in range(num_layers):
            if i < max_num_layer:
                self.layer_types.append("full attention")
            else:
                self.layer_types.append("sliding window attention")

        for i in range(num_layers):
            if self.layer_types[i] == "full attention":
                attention_layer = SlidingWindowLayer(self.hidden_size, self.num_heads, None)
            else:
                attention_layer = SlidingWindowLayer(self.hidden_size, self.num_heads, self.sliding_window)
            self.layers.append(attention_layer)

        self.norm = RMSNorm(hidden_size)
        self.lm_head = torch.nn.Linear(hidden_size, hidden_size, bias=False)

if __name__ == "__main__":
    import torch
    hidden_size = 2
    test = RMSNorm(hidden_size)
    x = torch.rand((2, hidden_size))
    print(test(x))
    print("this is a test")

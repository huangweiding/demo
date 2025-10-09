# encoding=utf-8
######### Sliding Window Attention ##############
#########################

import torch

class RMSNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.weights = torch.nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True)+self.eps)
        return x*norm*self.weights


class MHA(torch.nn.Module):
    def __init__(self, hidden_size, num_heads=8, num_layers=16, sliding_window_size=None):
        assert hidden_size % num_heads == 0
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.sliding_window_size = sliding_window_size

        self.q_proj = torch.nn.Linear(hidden_size, hidden_size)
        self.k_proj = torch.nn.Linear(hidden_size, hidden_size)
        self.v_proj = torch.nn.Linear(hidden_size, hidden_size)
        self.o_proj = torch.nn.Linear(hidden_size, hidden_size)

    def forward(self, x, attention_mask=None):
        batch_size, seq_length, hidden_size = x.shape


        causal_mask = torch.triu(torch.ones((seq_length, seq_length)), diagonal=1).bool()

        if self.sliding_window_size is not None:
            causal_mask = self.apply_sliding_window(self.sliding_window_size, causal_mask)

        # create a causal mask
        if attention_mask is not None:
            # attention_mask [seq_length, seq_length] bool
            padding_mask = attention_mask

            causal_mask = padding_mask & causal_mask

        Q = self.q_proj(x).view(batch_size, seq_length, self.num_heads, hidden_size//self.num_heads).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_length, self.num_heads, hidden_size//self.num_heads).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_length, self.num_heads, hidden_size//self.num_heads).transpose(1, 2)

        attention_score = torch.matmul(Q, K.transpose(-1, -2))  # Proper transpose for K
        
        # Scale attention scores
        attention_score = attention_score / (hidden_size // self.num_heads) ** 0.5

        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        attention_score = attention_score.masked_fill(~causal_mask, -1e9)
        attention_weights = torch.nn.functional.softmax(attention_score, dim=-1)

        # [batch_size, num_heads, seq_length, seq_length] [batch_size, num_heads, seq_length, head_dim]
        output = torch.matmul(attention_weights, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_length, -1)

        output = self.o_proj(output)
        return attention_weights, output

    def apply_sliding_window(self, window_size, causal_mask):
        seq_length = causal_mask.size(0)
        causal_mask = torch.zeros((seq_length, seq_length)).bool()
        for i in range(len(causal_mask)):
            start_idx = max(0, i-window_size+1)
            causal_mask[i, start_idx:i+1] = True
        return causal_mask

        
if __name__ == "__main__":
    # Test sliding window attention
    mha_window = MHA(hidden_size=64, num_heads=4, sliding_window_size=3)
    
    print("Testing sliding window attention (window_size=3):")
    x = torch.randn(1, 6, 64)  # seq_len=6, hidden=64
    
    attn_weights_window, output_window = mha_window(x)
    print(f"Attention weights shape: {attn_weights_window.shape}")
    print(f"Output shape: {output_window.shape}")
    
    # Visualize attention pattern for first head
    print("\nAttention pattern for first sample, first head:")
    attention_pattern = attn_weights_window[0, 0]  # [seq_len, seq_len]
    
    print("Position -> attended positions:")
    for i in range(attention_pattern.size(0)):
        nonzero_indices = (attention_pattern[i] > 1e-5).nonzero().flatten()
        print(f"  {i}: {nonzero_indices.tolist()}")
    
    print("\nLet's verify sliding window constraint:")
    print(f"Position 4 attending to position 0: {attention_pattern[4, 0]:.6f} (should be ~0)")
    print(f"Position 5 attending to position 0: {attention_pattern[5, 0]:.6f} (should be ~0)")
    print(f"Position 5 attending to position 1: {attention_pattern[5, 1]:.6f} (should be ~0)")
    print(f"Position 5 attending to position 2: {attention_pattern[5, 2]:.6f} (should be >0)")
    
    # Show the generated mask
    print("\nGenerated sliding window mask:")
    test_mask = torch.triu(torch.ones((6, 6)), diagonal=1).bool()
    sliding_window_mask = mha_window.apply_sliding_window(3, test_mask)
    print("Mask visualization (True=allow attention):")
    print(sliding_window_mask.int())

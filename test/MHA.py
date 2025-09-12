import math
import torch

class MHA(torch.nn.Module):
    def __init__(self, num_heads, hidden_size) -> None:
        super().__init__()
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.q_proj = torch.nn.Linear(hidden_size, hidden_size)
        self.k_proj = torch.nn.Linear(hidden_size, hidden_size)
        self.v_proj = torch.nn.Linear(hidden_size, hidden_size)
        self.o_proj = torch.nn.Linear(hidden_size, hidden_size)

    def forward(self, Q, K, V, attention_mask=None):
        batch_size, seq_length, hidden_size = Q.size()
        Q = self.q_proj(Q).view(batch_size, seq_length, self.num_heads, hidden_size//self.num_heads).transpose(1, 2)
        K = self.k_proj(K).view(batch_size, seq_length, self.num_heads, hidden_size//self.num_heads).transpose(1, 2)
        V = self.v_proj(V).view(batch_size, seq_length, self.num_heads, hidden_size//self.num_heads).transpose(1, 2)
        output, attention_weights = self.apply_attention(Q, K, V, attention_mask=attention_mask)

    def apply_attention(self, Q, K, V, attention_mask=None):
        # Q size [batch_size, num_heads, seq_length, head_dims]
        batch_size, num_heads, seq_length, head_dims = Q.size()
        if attention_mask is None:
            pass
        # create a causal mask
        reversed_causal_mask = torch.triu(torch.ones((seq_length, seq_length)), diagonal=1).bool()
        causal_mask = ~reversed_causal_mask

        attention_weights = torch.matmul(Q, K.transpose(2, 3))
        attention_weights /= math.sqrt(head_dims)
        attention_weights = attention_weights.masked_fill(~causal_mask, value=float(-math.inf))

        attention_weights = torch.nn.functional.softmax(attention_weights, dim=-1)
        

        return 0, attention_weights






        


if __name__ == "__main__":
    Q = torch.rand((2, 5, 8))
    K = torch.rand((2, 5, 8))
    V = torch.rand((2, 5, 8))
    mha = MHA(2, 8)
    res = mha(Q, K, V)
    print(res)


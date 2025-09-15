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

        output = output.transpose(1, 2).contiguous().view(batch_size, seq_length, -1)

        output = self.o_proj(output)

        return output, attention_weights

    def apply_attention(self, Q, K, V, attention_mask=None):
        # Q size [batch_size, num_heads, seq_length, head_dims]
        batch_size, num_heads, seq_length, head_dims = Q.size()

        reversed_causal_mask = torch.triu(torch.ones((seq_length, seq_length)), diagonal=1).bool()
        causal_mask = ~reversed_causal_mask
        if attention_mask is not None:
            # create a causal mask
            # padding_mask = [BS, seq_length]
            padding_mask = attention_mask.unsqueeze(1).unsqueeze(1).bool() #[BS, 1, 1, seq_length]
            transpose_padding_mask = padding_mask.transpose(2, 3)
            key_padding_mask = padding_mask & transpose_padding_mask
            causal_mask = causal_mask & padding_mask

        attention_weights = torch.matmul(Q, K.transpose(2, 3))
        attention_weights /= math.sqrt(head_dims)
        attention_weights = attention_weights.masked_fill(~causal_mask, value=float(-math.inf))

        attention_weights = torch.nn.functional.softmax(attention_weights, dim=-1)

        output = torch.matmul(attention_weights, V)
        

        return output, attention_weights






        


if __name__ == "__main__":
    Q = torch.rand((2, 5, 8))
    K = torch.rand((2, 5, 8))
    V = torch.rand((2, 5, 8))
    mha = MHA(2, 8)
    attention_mask = torch.ones((2, 5))
    attention_mask[0][3:] = 0
    attention_mask[1][4:] = 0
    res = mha(Q, K, V, attention_mask=attention_mask)
    print(res)


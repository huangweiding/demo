import torch
import math

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

def create_attention_mask(seq_length, sliding_window):
    if sliding_window is None:
        causal_mask = torch.triu(torch.ones((seq_length, seq_length)), diagonal=1).bool()
        attention_mask = ~causal_mask
    else:
        # sliding window is not None
        causal_mask = torch.triu(torch.ones((seq_length, seq_length)), diagonal=1).bool()
        sliding_window_matrix = torch.zeros((seq_length, seq_length), dtype=torch.bool)
        for i in range(seq_length):
            start_idx = max(0, i-sliding_window+1)
            sliding_window_matrix[i, start_idx:i+1] = True

        attention_mask = ~causal_mask & sliding_window_matrix
    return attention_mask
        



def apply_sliding_window_attention(Q, K, V, sliding_window, mask=None, padding_mask=None):

    batch_size, num_heads, seq_length, head_dim = Q.shape

    attention_weights = torch.matmul(Q, K.transpose(2, 3))/math.sqrt(head_dim)
    if sliding_window is None:
        # full attention
        attention_mask = create_attention_mask(seq_length, sliding_window)
        attention_mask = attention_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, num_heads, -1, -1)
    elif mask is None:
        # sliding window attention
        attention_mask = create_attention_mask(seq_length, sliding_window)
        attention_mask = attention_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, num_heads, -1, -1)
    else:
        attention_mask = mask

    if padding_mask is not None and sliding_window:
        # padding mask [BS, seq_length]
        padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)
        # [BS, 1, 1, seq_length]
        transpose_padding_mask = padding_mask.transpose(-2, -1)
        #[BS, 1, seq_length, 1]
        final_padding_mask = padding_mask & transpose_padding_mask

        attention_mask = final_padding_mask & attention_mask

    attention_weights = attention_weights.masked_fill(~attention_mask, value=float('-inf'))

    attention_weights = torch.nn.functional.softmax(attention_weights, dim=-1)

    # attention_weights [batch_size, num_heads, seq_length, seq_length]
    # v [batch_size, num_heads, seq_length, head_dim]
    # result [batch_size, num_heads, seq_length, head_dim]

    output = torch.matmul(attention_weights, V)

    return output, attention_weights

    


class SlidingWindowLayer(torch.nn.Module):
    def __init__(self, hidden_size, num_heads, sliding_window, mask=None):
        assert hidden_size % num_heads == 0
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.sliding_window = sliding_window
        self.mask = mask

        self.q_proj = torch.nn.Linear(hidden_size, hidden_size)
        self.k_proj = torch.nn.Linear(hidden_size, hidden_size)
        self.v_proj = torch.nn.Linear(hidden_size, hidden_size)
        self.o_proj = torch.nn.Linear(hidden_size, hidden_size)

        self.q_norm = RMSNorm(hidden_size)
        self.k_norm = RMSNorm(hidden_size)




    def forward(self, inputs, attention_mask=None):
        
        batch_size, seq_length, hidden_size = inputs.shape

        Q = self.q_proj(inputs).view(batch_size, seq_length, self.num_heads, self.hidden_size//self.num_heads).transpose(1, 2)
        K = self.k_proj(inputs).view(batch_size, seq_length, self.num_heads, self.hidden_size//self.num_heads).transpose(1, 2)
        V = self.v_proj(inputs).view(batch_size, seq_length, self.num_heads, self.hidden_size//self.num_heads).transpose(1, 2)

        output, attention_weights = apply_sliding_window_attention(Q, K, V, self.sliding_window, padding_mask=attention_mask)


        output = output.transpose(1, 2).contiguous().view(batch_size, seq_length, -1)

        output = self.o_proj(output)

        return output, attention_weights


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
        self.lm_head = torch.nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, x, attention_mask=None):

        input_embedding = self.embedding_layer(x)
        hidden_states = input_embedding
        for i in range(len(self.layers)):
            hidden_states, _ = self.layers[i](hidden_states, attention_mask=attention_mask)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits

if __name__ == "__main__":
    vocab_size = 1000
    hidden_size = 512
    num_layers = 6
    num_heads = 8
    sliding_window = 4
    max_window_layers = 3  # 前3层使用全注意力，后3层使用滑动窗口注意力
    
    # 创建模型
    model = MyModel(
        vocab_size, hidden_size, num_heads, num_layers, sliding_window, max_window_layers
    )
    
    # 创建输入
    batch_size = 2
    seq_len = 8
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

    input_mask = input_ids.clone()

    input_mask[0, -1:] = 0
    input_mask[1, -2:] = 0
    input_mask = torch.greater(input_mask, 0)
    
    print(f"输入形状: {input_ids.shape}")
    print(f"滑动窗口大小: {sliding_window}")
    print(f"最大窗口层数: {max_window_layers}\n")
    
    # 前向传播
    with torch.no_grad():
        logits = model(input_ids, attention_mask=input_mask)
    print(logits)

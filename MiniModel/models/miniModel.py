import torch
from typing import List
from transformers import PretrainedConfig, PretrainedModel, GenerationMixin

THETA = 10000

class MiniConfig(PretrainedConfig):
    model_type = "MiniModel"
    def __init__(self, 
                 vocab_size: int, 
                 embedding_size: int=2048,
                 dropout_rate: float=0.1,
                 num_heads: int=8,
                 max_length: int=2048,
                 eps: float=1e-6,
                 **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.dropout_rate = dropout_rate
        self.num_heads = num_heads
        self.max_length = max_length
        self.eps = eps



def relative_positional_embedding(seq_length: int, dim: int):
    freq = 1/THETA **(torch.arange(0, dim, 2)/dim)
    positions = torch.arange(0, seq_length)

    r = torch.outer(positions, freq)

    sin = r.sin()
    cos = r.cos()
    freq_sin = torch.cat([sin, sin], dim=-1)
    freq_cos = torch.cat([cos, cos], dim=-1)

    return freq_sin, freq_cos

def apply_rotary_positional_embedding(q, k, cos, sin, unsqueeze_dim=1):
    def rotate_half(x):
        return torch.cat([-x[..., :x.size(-1)//2], x[..., x.size(-1)//2:]])
    q_embed = q*cos.unsqueeze(unsqueeze_dim) + rotate_half(q)*sin.unsqueeze(unsqueeze_dim)
    k_embed = k*cos.unsqueeze(unsqueeze_dim) + rotate_half(k)*sin.unsqueeze(unsqueeze_dim)
    return q_embed, k_embed


class RMSNorm(torch.nn.Module):
    def __init__(self, hidden_size: int, eps: float=1e-5):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))

    def _norm(self, inputs):
        return inputs * torch.rsqrt(inputs.pow(2).mean(dim=-1, keepdim=True)+self.eps)

    def forward(self, inputs):
        return self.weight * self._norm(inputs.float()).type_as(inputs)


class Embedding(torch.nn.Module):
    def __init__(self, vocab_size: int, embedding_size: int, max_length: int=2048, padding_idx: int=0, dropout_rate: float=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.max_length = max_length
        self.padding_idx = padding_idx
        self.dropout_rate = dropout_rate
        self.embedding = torch.nn.Embedding(vocab_size, embedding_size, padding_idx=padding_idx)

        self.dropout = torch.nn.Dropout(dropout_rate)

        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        """加速收敛"""
        torch.nn.init.normal_(self.embedding.weight, mean=0, std=self.embedding_size**-0.5)
        if self.padding_idx is not None:
            torch.nn.init.constant_(self.embedding.weight[self.padding_idx], 0)

    def forward(self, inputs, attention_mask=None):

        embedding_state = self.embedding(inputs)

        final_embedding_state = self.dropout(embedding_state)

        return final_embedding_state

class Attention(torch.nn.Module):
    def __init__(self, config: MiniConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.hidden_size = config.hidden_size
        assert self.hidden_size % self.num_heads == 0
        self.Q = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.K = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.V = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.O = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=False)

    def forward(self, inputs, position_embedding, attention_mask=None):

        cos, sin = position_embedding
        batch_size, seq_length, hidden_size = inputs.size()


        q = self.Q(inputs).view(batch_size, seq_length, self.num_heads, self.hidden_size//self.num_heads).transpose(2, 3)
        k = self.K(inputs).view(batch_size, seq_length, self.num_heads, self.hidden_size//self.num_heads).transpose(2, 3)
        v = self.V(inputs).view(batch_size, seq_length, self.num_heads, self.hidden_size//self.num_heads).transpose(2, 3)

        q, k = apply_rotary_positional_embedding(q, k, cos[:seq_length], sin[:seq_length])

        attention_weights = torch.matmul(q, k)

        if attention_mask is None:
            # if attention_mask is None, we use causal_attention
            attention_mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1).bool()

        attention_weights = attention_weights.masked_fill(attention_mask, -1e9)

        attention_weights = torch.nn.functional.softmax(attention_weights)

        output = torch.matmul(attention_weights, v)

        output = output.transpose(2, 3).view(batch_size, seq_length, -1)

        output = self.O(output)

        return attention_weights, output


class MiniBlock(torch.nn.Module):
    def __init__(self, layer_id: int, config: MiniConfig):
        self.num_heads = config.num_heads
        self.hidden_size = config.hidden_size
        self.attn = Attention(config)
        self.layer_id = layer_id

        self.input_layernorm = RMSNorm(config.hidden_size, config.eps)
        self.post_layernorm = RMSNorm(config.hidden_size, config.eps)


    def forward(self, inputs, position_embedding, attention_mask=None):
        # inputs dim
        # [batch_size, seq_length, hidden_size]
        inputs = self.input_layernorm(inputs)

        attention_weights, output = self.attn(inputs, position_embedding)
        




class miniModel(torch.nn.Module):
    def __init__(self, config: MiniConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.embedding_size = config.embedding_size
        self.num_Heads = config.num_heads
        self.layer_norm = RMSNorm(config.hidden_size, eps=config.eps)
        self.layer_num = config.layer_num

        self.miniBlocks = torch.nn.ModuleList([MiniBlock(l, config) for l in range(self.layer_num)])

        freqs_cos, freqs_sin = relative_positional_embedding(seq_length=config.max_length,
                                                             dim=config.hidden_size//config.num_heads)

        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self, inputs):


class miniModelForCausalLM(PretrainedModel, GenerationMixin):
    def __init__(self, config: MiniConfig):
        self.config = config
        self.model = miniModel(config)

    def forward(self, inputs, attention_mask=None):
        pass

if __name__ == "__main__":
    seq_length = 2000
    dim = 256
    relative_positional_embedding(seq_length, dim)






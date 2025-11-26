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
                 hidden_size: int=2048,
                 layer_num: int=8,
                 **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.dropout_rate = dropout_rate
        self.num_heads = num_heads
        self.max_length = max_length
        self.eps = eps
        self.hidden_size = hidden_size
        self.layer_num = layer_num



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
    def __init__(self, vocab_size: int, embedding_size: int, padding_idx: int=0, dropout_rate: float=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
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

        self.dropout = torch.nn.Dropout(config.dropout_rate)

    def forward(self, 
                inputs, 
                position_embedding, 
                past_key_values=None, 
                use_cache=False,
                attention_mask=None):

        cos, sin = position_embedding
        batch_size, seq_length, hidden_size = inputs.size()


        q = self.Q(inputs).view(batch_size, seq_length, self.num_heads, self.hidden_size//self.num_heads).transpose(2, 3)
        k = self.K(inputs).view(batch_size, seq_length, self.num_heads, self.hidden_size//self.num_heads).transpose(2, 3)
        v = self.V(inputs).view(batch_size, seq_length, self.num_heads, self.hidden_size//self.num_heads).transpose(2, 3)

        q, k = apply_rotary_positional_embedding(q, k, cos[:seq_length], sin[:seq_length])

        if past_key_values is not None:
            k = torch.cat([past_key_values[0], k], dim=1)
            v = torch.cat([past_key_values[1], v], dim=1)
        past_kv = (k, v) if use_cache else None


        attention_weights = torch.matmul(q, k)

        if attention_mask is None:
            # if attention_mask is None, we use causal_attention
            attention_mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1).bool()

        attention_weights = attention_weights.masked_fill(attention_mask, -1e9)

        attention_weights = torch.nn.functional.softmax(attention_weights)

        output = torch.matmul(attention_weights, v)

        output = output.transpose(2, 3).view(batch_size, seq_length, -1)

        output = self.O(output)
        output = self.dropout(output)

        return output, past_kv

class feedForwardBlock(torch.nn.Module):
    def __init__(self, config: MiniConfig):
        self.hidden_size = config.hidden_size
        if config.intermediate_size is None:
            intermediate_size = int(config.hidden_size * 8 / 3)
            config.intermediate_size = 64*((intermediate_size+64-1)//64)
        self.up_proj = torch.nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.gate_proj = torch.nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = torch.nn.Linear(config.intermedaite_size, config.hidden_size, bias=False)
        self.dropout = torch.nn.Dropout(config.dropout_rate)
        self.act_fn = torch.nn.functional.silu

    def forward(self, inputs):
        return self.dropout(self.down_proj(self.act_fn(self.gate_proj(inputs))*self.up_proj(inputs)))


class MiniBlock(torch.nn.Module):
    def __init__(self, layer_id: int, config: MiniConfig):
        self.num_heads = config.num_heads
        self.hidden_size = config.hidden_size
        self.attn = Attention(config)
        self.layer_id = layer_id

        self.input_layernorm = RMSNorm(config.hidden_size, config.eps)
        self.post_layernorm = RMSNorm(config.hidden_size, config.eps)
        self.mlp = feedForwardBlock(config)


    def forward(self,
                inputs,
                position_embedding,
                past_key_values=None,
                attention_mask=None,
                use_cache=False):
        # inputs dim
        # [batch_size, seq_length, hidden_size]
        #
        residual = inputs
        inputs = self.input_layernorm(inputs)

        output, past_kv = self.attn(inputs,
                                    position_embedding,
                                    past_key_values, 
                                    use_cache,
                                    attention_mask=attention_mask)

        hidden_states = residual + output
        hidden_states = hidden_states + self.mlp(self.post_layernorm(hidden_states))

        return hidden_states, past_kv

class miniModel(torch.nn.Module):
    def __init__(self, config: MiniConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.embedding_size = config.embedding_size
        self.num_Heads = config.num_heads
        self.layer_norm = RMSNorm(config.hidden_size, eps=config.eps)
        self.layer_num = config.layer_num


        self.embedding_layer = Embedding(config.vocab_size, config.embedding_size)
        self.miniBlocks = torch.nn.ModuleList([MiniBlock(l, config) for l in range(self.layer_num)])


        freqs_cos, freqs_sin = relative_positional_embedding(seq_length=config.max_length,
                                                             dim=config.hidden_size//config.num_heads)

        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self, 
                inputs, 
                past_key_values=None,
                use_cache=False,
                attention_mask=None
                ):

        batch_size, seq_length = inputs.shape
        past_key_values = past_key_values or [None] * self.layer_num

        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        position_embeddings = (self.freqs_cos[start_pos: start_pos+seq_length],
                           self.freqs_sin[start_pos: start_pos+seq_length])

        hidden_states = self.Embedding(inputs)

        presents = []
        for layer_idx, (layer, past_key_values) in enumerate(zip(self.miniBlocks, past_key_values)):
            output, present = layer(hidden_states, position_embeddings, past_key_values, use_cache, attention_mask)
            presents.append(present)

        hidden_states = self.layer_norm(hidden_states)

        return hidden_states










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






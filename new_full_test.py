import torch
import math

class RMSNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        self.weights = torch.nn.Parameter(torch.ones((hidden_size, hidden_size)))
        self.eps = eps

    def forward(self, inputs, attention_mask=None):
        rms_norm = torch.rsqrt(inputs.pow(2).mean(-1, keepdim=True)+self.eps)
        return inputs * rms_norm* self.weights

class SimpleExpert(torch.nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.hidden_size=hidden_size
        self.intermediate_size = intermediate_size

        self.down_proj = torch.nn.Linear(intermediate_size, hidden_size)
        self.gate = torch.nn.Linear(hidden_size, intermediate_size)
        self.up_proj = torch.nn.Linear(hidden_size, intermediate_size)
        self.act_fn = torch.nn.functional.silu

    def forward(self, inputs, attention_mask=None):
        return self.down_proj(self.act_fn(self.gate(inputs))*self.up_proj(inputs))

class MoeBlock(torch.nn.Module):
    def __init__(self, hidden_size, moe_intermediate_size, num_experts, num_expert_per_topk):
        super().__init__()
        self.hidden_size = hidden_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_experts = num_experts
        self.num_expert_per_topk = num_expert_per_topk

        self.expert_layers = torch.nn.ModuleList([SimpleExpert(hidden_size, moe_intermediate_size) for _ in range(num_experts)])
        self.gate = torch.nn.Linear(hidden_size, num_experts)

    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_length, hidden_size = hidden_states.shape

        hidden_states = hidden_states.view(-1, hidden_size)
        # hidden_size[bs*seq_length, hidden_states]
        expert_prob = self.gate(hidden_states)
        # selected_experts [BS*seq_length, num_expert]
        expert_prob = torch.nn.functional.softmax(expert_prob, dim=-1)
        router_weights, selected_experts = torch.topk(expert_prob, self.num_expert_per_topk, dim=-1)

        # selected_agents [BS*seq_length, num_expert_per_topk] -> [BS*seq_length, num_expert_per_topk, num_experts] one host selected_agents
        expert_mask = torch.nn.functional.one_hot(selected_experts, self.num_experts)
        expert_mask = expert_mask.permute(2, 1, 0)
        # expert_mask -> [num_expert, num_expert_per_topk, bs*seq_length]
        
        hitted_experts = torch.greater(expert_mask.sum(dim=(-2, -1)), 0).nonzero()

        initial_hidden_states = torch.zeros(batch_size*seq_length, hidden_size, dtype=hidden_states.dtype, device=hidden_states.device)
        for expert_idx in hitted_experts:
            expert_layer = self.expert_layers[expert_idx]
            # expert_mask [num_experts, num_expert_per_topk, batch_size*seq_length]
            # expert_mask[None, expert_idx] -> expert[1, num_expert_per_topk, batch_size*seq_length]
            
            x, y = torch.where(expert_mask[None, expert_idx].squeeze(0).view(-1, batch_size*seq_length))
            if y.numel() > 0:
                selected_hidden_states = hidden_states[None, y].view(-1, hidden_size)
                current_hidden_states = expert_layer(selected_hidden_states) * router_weights[y, x, None]
                # router_weights is [bs*seq_length, num_expert_per_topk] so we use y, x as y represents the idx of bs*seq_length, x is the idx of chosen expert

                initial_hidden_states.index_add_(0, y, current_hidden_states)

        initial_hidden_states = initial_hidden_states.view(batch_size, seq_length, hidden_size)

        return initial_hidden_states


            

        







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
        residual = inputs

        Q = self.q_proj(inputs).view(batch_size, seq_length, self.num_heads, -1).transpose(-2, -1)
        K = self.k_proj(inputs).view(batch_size, seq_length, self.num_heads, -1).transpose(-2, -1)
        V = self.v_proj(inputs).view(batch_size, seq_length, self.num_heads, -1).transpose(-2, -1)

        Q = self.q_norm(Q)
        K = self.k_norm(K)

        final_attention_mask = create_attention_mask(seq_length, self.sliding_window, mask=attention_mask)

        scores, attention_weights = apply_attention(Q, K, V, attention_mask=final_attention_mask)

        scores = scores.transpose(1, 2).contiguous().view(batch_size, seq_length, -1)
        scores = self.o_proj(scores)

        hidden_states = residual + scores

        return scores, attention_weights


def apply_attention(Q, K, V, attention_mask=None):
    batch_size, num_heads, seq_length, head_dim = Q.shape

    attention_weights = torch.matmul(Q, K.tranpose(-2, -1))/math.sqrt(head_dim)
    # divide by sqrt(head_dim) to keep std at 1

    if attention_mask is not None:
        attention_weights = attention_weights.masked_fill(~attention_mask, value=-float('inf'))
        # masked fill fills where the matrix is True
    attention_weights = torch.nn.functional.softmax(attention_weights)

    # attention_weights size [BS, num_heads, seq_length, head_dim]
    scores = torch.matmul(attention_weights, V)
    return scores, attention_weights



def create_attention_mask(seq_length, sliding_window, mask=None):
    # sliding window : sliding window size
    # mask: padding mask
    # padding mask: [BS, seq_length] bool

    causal_mask_inversed = torch.triu(torch.ones((seq_length, seq_length)), diagonal=1).bool()
    if sliding_window is not None:
        initial_mask = torch.zeros(seq_length, seq_length)
        for i in range(seq_length):
            start_idx = max(0, i-sliding_window+1)
            initial_mask[i][start_idx:i+1] = 1
        initial_mask = initial_mask.bool()
        attention_mask = initial_mask & ~causal_mask_inversed
    else:
        attention_mask = ~causal_mask_inversed

    final_attention_mask = attention_mask.unsqueeze(0).unsqueeze(1)

    if mask is not None:
        mask = mask.unsqueeze(1).unsqueeze(2)
        transposed_mask = mask.transpose(-2, -1)
        padding_mask = mask & transposed_mask
        # padding mask size [BS, 1, seq_length, seq_length]
        # we need to merge with attention_mask whose size is [seq_length, seq_length]
        final_attention_mask = final_attention_mask & padding_mask
    return final_attention_mask


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

if __name__ == "__main__":
    # mask = torch.ones((2, 5))
    # mask[0][-1:] = 0
    # mask[1][-2:] = 0
    # mask = mask.bool()
    # breakpoint()
    # create_attention_mask(5, 2, mask=mask)
    import torch
    test = MoeBlock(512, 1024, 8, 2)
    hidden_states = torch.rand(2, 8, 512)
    print(test(hidden_states))


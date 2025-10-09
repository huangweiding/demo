import torch
class RMSNorm(torch.nn.Module):
    def __init__(self):
        super().__init__()

class SimpleExpert(torch.nn.Module):
    def __init__(self, hidden_size, intermediate_size=2048):
        super().__init__()
        self.hidden_size = hidden_size
        self.act_fn = torch.nn.functional.silu
        self.up_proj = torch.nn.Linear(hidden_size, intermediate_size)
        self.gate_proj = torch.nn.Linear(hidden_size, intermediate_size)
        self.down_proj = torch.nn.Linear(intermediate_size, hidden_size)

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)*self.up_proj(x)))


class MOEBlock(torch.nn.Module):
    def __init__(self, num_experts, num_experts_per_topk, hidden_size, intermediate_size):
        super().__init__()
        self.num_experts = num_experts
        self.num_experts_per_topk = num_experts_per_topk
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.expert_layers = torch.nn.ModuleList([SimpleExpert(hidden_size, intermediate_size) for _ in range(num_experts)])

        self.gate_proj = torch.nn.Linear(hidden_size, num_experts)

    def forward(self, x, attention_mask=None):
        batch_size, seq_length, hidden_size = x.shape

        hidden_states = x.view(-1, hidden_size) # [batch_size * seq_length, hidden_size]

        selected_experts = self.gate_proj(hidden_states) #[batch_size*seq_length, num_experts]

        selected_experts = torch.nn.functional.softmax(selected_experts, dim=-1)

        router_weights, selected_experts = torch.topk(selected_experts, self.num_experts_per_topk)

        expert_mask = torch.nn.functional.one_hot(selected_experts, self.num_experts) # [batch_size*seq_length, num_experts_per_topk, one hot of selected_expert_index in num_experts]
        expert_mask = expert_mask.permute(2,1,0) #[8, 2, batch_size*seq_length]


        empty_hidden_states = torch.zeros((batch_size*seq_length, hidden_size), dtype=x.dtype)

        chosen_experts = torch.greater(expert_mask.sum((-2, -1)), 0).nonzero()

        for expert_idx in chosen_experts:
            # chosen expert layer
            expert_layer = self.expert_layers[expert_idx]
            p_x, p_y = torch.where(expert_mask[expert_idx].view(-1, batch_size*seq_length).squeeze(0))

            if p_y.numel() > 0:
                current_hidden_states = expert_layer(hidden_states[None, p_y]).squeeze(0) * router_weights[p_y,p_x].unsqueeze(-1)
                empty_hidden_states.index_add_(0, p_y, current_hidden_states)
        empty_hidden_states = empty_hidden_states.view(batch_size, seq_length, hidden_size)
        return empty_hidden_states


if __name__ == "__main__":
    num_experts = 8
    num_experts_per_topk = 2
    hidden_size = 256
    intermediate_size = 2048
    x = torch.normal(0, 0.1, (2, 8, 256))
    moe_block = MOEBlock(num_experts, num_experts_per_topk, hidden_size, intermediate_size)
    b = moe_block(x)
    print(b)


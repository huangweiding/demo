import torch
class SimpleExpert(torch.nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.up = torch.nn.Linear(hidden_size, intermediate_size)
        self.down = torch.nn.Linear(intermediate_size, hidden_size)
        self.gate = torch.nn.Linear(hidden_size, intermediate_size)
        self.act_fn = torch.nn.functional.silu

    def forward(self, inputs):
        return self.down(self.act_fn(self.gate(inputs))* self.up(inputs))

class SimpleMoeBlock(torch.nn.Module):
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

        selected_experts = self.gate(hidden_states)
        # [bs*seq_length, hidden_size]
        selected_experts = torch.nn.functional.softmax(selected_experts, dim=-1)
        router_weights, selected_experts = torch.topk(selected_experts, self.num_expert_per_topk, dim=-1)
        # selected_experts

        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
        # expert_mask [num_expert, num_expert_per_topk, bs*seq_length]
        hitted_experts = torch.greater(expert_mask.sum(dim=(-2, -1)), 0).nonzero()

        final_hidden_states = torch.zeros((batch_size*seq_length, hidden_size), dtype=hidden_states.dtype)
        for expert_idx in hitted_experts:
            expert_layer = self.expert_layers[expert_idx]
            x, y = torch.where(expert_mask[None, expert_idx].view(-1, batch_size*seq_length).squeeze(0))

            if y.numel() > 0:
                # hitted experts is not empty
                current_hidden_states = expert_layer(hidden_states[None, y].view(-1, hidden_size)).squeeze(0) * router_weights[y, x].unsqueeze(-1)
                final_hidden_states.index_add_(0, y, current_hidden_states)

        final_hidden_states = final_hidden_states.view(batch_size, seq_length, hidden_size)

        return final_hidden_states


if __name__ == "__main__":
    hidden_size = 256
    moe_intermediate_size = 1025
    num_experts = 8
    num_expert_per_topk = 2
    hidden_states = torch.rand((2, 8, 256))
    test = SimpleMoeBlock(hidden_size, moe_intermediate_size, num_experts, num_expert_per_topk)
    a = test(hidden_states)
    print(a)



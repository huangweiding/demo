import torch

class SimpleExpert(torch.nn.Module):
    def __init__(self, hidden_size, intermediate_size, bias=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate = torch.nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up = torch.nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down = torch.nn.Linear(intermediate_size, hidden_size, bias=bias)

        self.act_fn = torch.nn.functional.silu

    def forward(self, x):
        return self.down(self.act_fn(self.gate(x))* self.up(x))

class SimpleMoeBlock(torch.nn.Module):
    def __init__(self, hidden_size, num_expert_per_topk, num_experts, moe_intermediate_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_expert_per_topk = num_expert_per_topk
        self.num_experts = num_experts
        self.moe_intermediate_size = moe_intermediate_size

        self.expert_layers = torch.nn.ModuleList([SimpleExpert(hidden_size, moe_intermediate_size) for _ in range(num_experts)])

        self.gate = torch.nn.Linear(hidden_size, num_experts, bias=False)

    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_length, hidden_size = hidden_states.shape

        hidden_states = hidden_states.view(-1, hidden_size) 
        # [bs*seq_length, hidden_size]

        router_logits = self.gate(hidden_states)

        router_weights = torch.nn.functional.softmax(router_logits, dim=-1)
        router_weights, selected_experts = torch.topk(router_weights, self.num_expert_per_topk)

        #initialize an all zero matrix 
        final_hidden_states = torch.zeros((batch_size*seq_length, hidden_size), dtype=hidden_states.dtype, device=hidden_states.device)


        # 需要确认哪些expert被用到了
        expert_mask = torch.nn.functional.one_hot(selected_experts, self.num_experts).permute(2, 1, 0)
        # [bs*seq_length, choosen_experts_per_topk, num_experts] permute into [num_expert, choose_experts_per_topk, bs*seq_elgnth]
        # 计算每一个expert，在哪一个位置(如果每个token选择两个expert，那就有0，1两个位置， 在哪一个token位置上被选中)

        hitted_experts = torch.greater(expert_mask.sum(dim=(-2, -1)), 0).nonzero()
        # 取出被用到的所有expert的index

        for expert_idx in hitted_experts:
            expert_layer = self.expert_layers[expert_idx]
            x_list, y_list = torch.where(expert_mask[None, expert_idx].view(-1, batch_size*seq_length).squeeze(0))
            # x_list, 是第一个expert 还是第二个expert, y_list 当前expert 是被哪一个token选中了

            if y_list.numel() > 0:
                #如果 y_list有非0值, 如果没有非0值，y_list会是空list []
                selected_hidden_states = hidden_states[None, y_list].view(-1, hidden_size)
                current_hidden_states = expert_layer(selected_hidden_states) * router_weights[y_list, x_list]
                #hidden_states 现在是二维的[bs*seq_length, hidden_size], 
                # router_weights 也是一个二维矩阵[bs*seq_length, num_expert_per_topk], 所以需要反过来指定位置 y_list 是bs*seq_length的维度， x_list是指选中的是第一个还是第二个expert

if __name__ == "__main__":
    import torch
    hidden_size = 256
    num_expert_per_topk = 2
    num_experts = 8
    moe_intermediate_size = 1024
    test = SimpleMoeBlock(hidden_size=hidden_size, num_expert_per_topk=num_expert_per_topk, num_experts=8, moe_intermediate_size=moe_intermediate_size)
    inputs = torch.rand((2, 8, 256))
    test(inputs)









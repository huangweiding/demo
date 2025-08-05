import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RMSNorm(nn.Module):
    """RMSNorm实现，与Qwen3保持一致"""
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
    
    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class SimpleExpert(nn.Module):
    """基于Qwen3架构的专家网络"""
    def __init__(self, hidden_size, intermediate_size, bias=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        
        # 三个线性投影层，与Qwen3保持一致
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)
        
        # 激活函数
        self.act_fn = F.silu
        
    def forward(self, x):
        # SwiGLU激活函数，与Qwen3MoeMLP完全一致
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class SimpleMoEBlock(nn.Module):
    """基于Qwen3架构的MoE块"""
    def __init__(self, hidden_size, num_experts, num_experts_per_tok, moe_intermediate_size, norm_topk_prob=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_intermediate_size = moe_intermediate_size
        self.norm_topk_prob = norm_topk_prob
        
        # 路由器，与Qwen3保持一致
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        
        # 专家网络
        self.experts = nn.ModuleList([
            SimpleExpert(hidden_size, moe_intermediate_size) 
            for _ in range(num_experts)
        ])
        
    def forward(self, hidden_states):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        
        # 重塑为 [batch*seq_len, hidden_dim]
        hidden_states = hidden_states.view(-1, hidden_dim)
        
        # 1. 计算路由logits
        router_logits = self.gate(hidden_states)  # [batch*seq_len, num_experts]
        
        # 2. 应用softmax并选择top-k专家
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.num_experts_per_tok, dim=-1)
        
        # 3. 可选的topk概率归一化（Qwen3特性）
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        
        # 4. 转换回输入数据类型
        routing_weights = routing_weights.to(hidden_states.dtype)
        
        # 5. 初始化输出
        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), 
            dtype=hidden_states.dtype, 
            device=hidden_states.device
        )
        
        # 6. 创建专家掩码（与Qwen3一致）
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
        # [batch_size, chosen_experts, num_experts] -> [num_experts, chosen_experts, batch_size]
        
        # 7. 计算每个专家的输出
        expert_hitted = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hitted:
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))
            
            if top_x.numel() > 0:
                # 获取对应的输入和权重
                current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
                current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]
                
                # 累加到最终输出
                final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        
        # 8. 重塑回原始形状
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        
        return final_hidden_states, router_logits


class SimpleMoELayer(nn.Module):
    """简化的MoE层，包含注意力机制和MoE"""
    def __init__(self, hidden_size, num_attention_heads, num_experts, num_experts_per_tok, moe_intermediate_size, norm_topk_prob=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        
        # 注意力机制
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # 归一化层（使用RMSNorm，与Qwen3一致）
        self.input_norm = RMSNorm(hidden_size, eps=1e-6)
        self.post_attention_norm = RMSNorm(hidden_size, eps=1e-6)
        
        # MoE块
        self.moe = SimpleMoEBlock(hidden_size, num_experts, num_experts_per_tok, moe_intermediate_size, norm_topk_prob)
        
    def forward(self, hidden_states, attention_mask=None):
        residual = hidden_states
        
        # 1. 输入归一化
        hidden_states = self.input_norm(hidden_states)
        
        # 2. 自注意力
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # 计算Q, K, V
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # 应用掩码（如果有）
        if attention_mask is not None:
            scores = scores + attention_mask
        
        # Softmax
        attention_weights = F.softmax(scores, dim=-1)
        
        # 应用注意力权重
        attn_output = torch.matmul(attention_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_dim)
        attn_output = self.o_proj(attn_output)
        
        # 残差连接
        hidden_states = residual + attn_output
        
        # 3. MoE块
        residual = hidden_states
        hidden_states = self.post_attention_norm(hidden_states)
        moe_output, router_logits = self.moe(hidden_states)
        hidden_states = residual + moe_output
        
        return hidden_states, router_logits


class SimpleMoEModel(nn.Module):
    """简化的MoE模型"""
    def __init__(self, vocab_size, hidden_size, num_layers, num_attention_heads, 
                 num_experts, num_experts_per_tok, moe_intermediate_size, norm_topk_prob=False):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        
        self.layers = nn.ModuleList([
            SimpleMoELayer(
                hidden_size, num_attention_heads, num_experts, 
                num_experts_per_tok, moe_intermediate_size, norm_topk_prob
            )
            for _ in range(num_layers)
        ])
        
        self.norm = RMSNorm(hidden_size, eps=1e-6)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        
    def forward(self, input_ids, attention_mask=None):
        hidden_states = self.embed_tokens(input_ids)
        
        all_router_logits = []
        
        for layer in self.layers:
            hidden_states, router_logits = layer(hidden_states, attention_mask)
            all_router_logits.append(router_logits)
        
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        
        return logits, all_router_logits


def create_causal_mask(seq_len, device):
    """创建因果掩码"""
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]


def load_balancing_loss(router_logits, num_experts, num_experts_per_tok, attention_mask=None):
    """
    计算负载均衡损失
    Args:
        router_logits: [num_layers, BS*seq_len, num_experts]
        attention_mask: [batch_size, seq_len] 或 None，用于识别有效token

    """
    if not router_logits:
        return 0.0
    
    # 计算每个专家的使用频率
    expert_usage = torch.zeros(num_experts, device=router_logits[0].device)
    total_tokens = 0
    
    for logits in router_logits:
        # 获取top-k专家
        routing_weights = F.softmax(logits, dim=1)
        _, selected_experts = torch.topk(routing_weights, num_experts_per_tok, dim=-1)
        
        # 统计专家使用情况
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=num_experts)
        # expert_mask [bs*seq_length, num_experts_per_tok, num_experts]

        # 如果有attention_mask，只计算有效token
        if attention_mask is not None:
            # attention_mask: [batch_size, seq_len] -> [batch_size * seq_len]
            valid_mask = attention_mask.view(-1).bool()
            # 只对有效token统计专家使用情况
            expert_usage += expert_mask[valid_mask].sum(dim=(0, 1))
            total_tokens += valid_mask.sum().item()
        else:
            # 没有mask时，使用所有token
            expert_usage += expert_mask.sum(dim=(0, 1))
            total_tokens += logits.shape[0]
    
    # 计算理想均匀分布
    ideal_usage = total_tokens * num_experts_per_tok / num_experts
    
    # 计算负载均衡损失（方差）
    load_balancing_loss = torch.var(expert_usage / ideal_usage)
    
    return load_balancing_loss


def demo_moe():
    """演示基于Qwen3架构的MoE"""
    # 模型参数（基于Qwen3-MoE配置）
    vocab_size = 1000
    hidden_size = 512
    num_layers = 4
    num_attention_heads = 8
    num_experts = 8
    num_experts_per_tok = 2
    moe_intermediate_size = 256
    norm_topk_prob = False  # Qwen3的norm_topk_prob参数
    
    # 创建模型
    model = SimpleMoEModel(
        vocab_size, hidden_size, num_layers, num_attention_heads,
        num_experts, num_experts_per_tok, moe_intermediate_size, norm_topk_prob
    )
    
    # 创建输入（模拟有padding的情况）
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # 创建padding掩码（模拟不同长度的序列）
    # 第一个序列：有效长度8，后面2个是padding
    # 第二个序列：有效长度6，后面4个是padding
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    attention_mask[0, 8:] = False  # 第一个序列的padding
    attention_mask[1, 6:] = False  # 第二个序列的padding
    
    print(f"Attention mask (True=有效token, False=padding):")
    for i in range(batch_size):
        print(f"  序列{i}: {attention_mask[i].tolist()}")
    
    # 创建因果掩码（用于注意力计算）
    causal_mask = create_causal_mask(seq_len, input_ids.device)
    
    print("=== 基于Qwen3架构的MoE模型演示 ===")
    print(f"模型参数:")
    print(f"  - 词汇表大小: {vocab_size}")
    print(f"  - 隐藏层大小: {hidden_size}")
    print(f"  - 层数: {num_layers}")
    print(f"  - 注意力头数: {num_attention_heads}")
    print(f"  - 专家数量: {num_experts}")
    print(f"  - 每token专家数: {num_experts_per_tok}")
    print(f"  - 专家中间层大小: {moe_intermediate_size}")
    print(f"  - 归一化topk概率: {norm_topk_prob}")
    
    # 前向传播
    model.eval()
    with torch.no_grad():
        logits, router_logits = model(input_ids, causal_mask)
    
    print(f"\n输出形状: {logits.shape}")
    print(f"路由器logits数量: {len(router_logits)}")
    
    # 计算负载均衡损失
    aux_loss = load_balancing_loss(router_logits, num_experts, num_experts_per_tok, attention_mask)
    print(f"负载均衡损失: {aux_loss.item():.6f}")
    
    # 分析专家使用情况
    print(f"\n=== 专家使用情况分析 ===")
    expert_usage = torch.zeros(num_experts)
    total_tokens = 0
    
    # 创建有效token掩码（用于演示，实际应该从attention_mask获取）
    valid_mask = torch.ones(batch_size * seq_len, dtype=torch.bool)
    
    for logits in router_logits:
        routing_weights = F.softmax(logits, dim=1)
        _, selected_experts = torch.topk(routing_weights, num_experts_per_tok, dim=-1)
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=num_experts)
        
        # 只统计有效token
        expert_usage += expert_mask[valid_mask].sum(dim=(0, 1))
        total_tokens += valid_mask.sum().item()
    
    ideal_usage = total_tokens * num_experts_per_tok / num_experts
    print(f"有效token数: {total_tokens}")
    print(f"理想均匀分布: {ideal_usage:.2f}")
    print(f"各专家使用次数:")
    for i, usage in enumerate(expert_usage):
        print(f"  专家{i}: {usage.item():.0f} ({usage.item()/ideal_usage:.2%})")
    
    return model, logits, router_logits


if __name__ == "__main__":
    model, logits, router_logits = demo_moe() 

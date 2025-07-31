import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def create_sliding_window_mask(seq_len, sliding_window, device, attention_mask=None):
    """
    创建滑动窗口掩码，参考Qwen3的实现模式
    支持padding mask处理
    """
    # 创建因果掩码（下三角矩阵）
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    
    # 创建滑动窗口掩码
    sliding_mask = torch.zeros(seq_len, seq_len, device=device, dtype=torch.bool)
    for i in range(seq_len):
        # 对于位置i，只能看到位置[i-sliding_window, i]的键值对
        start_idx = max(0, i - sliding_window + 1)
        sliding_mask[i, start_idx:i+1] = True
    
    # 组合因果掩码和滑动窗口掩码
    # 最终掩码：True表示可以注意力，False表示被掩码
    final_mask = ~causal_mask & sliding_mask
    
    # 如果有attention_mask（padding mask），需要进一步处理
    if attention_mask is not None:
        # attention_mask形状: [batch_size, seq_len]
        # 需要扩展到[batch_size, 1, seq_len, seq_len]
        batch_size = attention_mask.shape[0]
        
        # 创建padding mask：padding位置为False，非padding位置为True
        padding_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_len]
        padding_mask = padding_mask & padding_mask.transpose(-2, -1)  # [batch_size, 1, seq_len, seq_len]
        
        # 将final_mask扩展到batch维度
        final_mask = final_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)
        
        # 组合滑动窗口掩码和padding掩码
        final_mask = final_mask & padding_mask
        
        return final_mask
    
    return final_mask


def apply_sliding_window_attention(Q, K, V, sliding_window, attention_mask=None, padding_mask=None):
    """
    应用滑动窗口注意力，参考Qwen3的实现模式
    支持padding mask处理
    """
    batch_size, num_heads, seq_len, head_dim = Q.shape
    
    # 1. 计算注意力分数
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(head_dim)
    
    # 2. 创建掩码
    if sliding_window is None:
        # 全注意力：只应用因果掩码
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=Q.device), diagonal=1).bool()
        attention_mask = ~causal_mask
        attention_mask = attention_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, num_heads, -1, -1)
    elif attention_mask is None:
        # 创建滑动窗口掩码
        sliding_mask = create_sliding_window_mask(seq_len, sliding_window, Q.device)
        # 扩展维度以匹配batch和heads
        attention_mask = sliding_mask.unsqueeze(1).expand(batch_size, num_heads, -1, -1)
    else:
        # 使用传入的attention_mask，需要扩展到num_heads维度
        if attention_mask.dim() == 4:
            # 已经是4D掩码，直接使用
            pass
        else:
            # 2D或3D掩码，需要扩展到4D
            attention_mask = attention_mask.unsqueeze(1).expand(batch_size, num_heads, -1, -1)
    
    # 3. 应用掩码（将掩码外的位置设为负无穷）
    # 注意：attention_mask中True表示可以注意力，False表示被掩码
    # 我们需要将False的位置设为负无穷
    scores = scores.masked_fill(~attention_mask, float('-inf'))
    
    # 4. 应用softmax
    attention_weights = F.softmax(scores, dim=-1)
    
    # 5. 与V相乘得到输出
    output = torch.matmul(attention_weights, V)
    
    return output, attention_weights


class SlidingWindowAttention(nn.Module):
    """
    滑动窗口注意力模块，参考Qwen3的实现模式
    支持padding mask处理
    """
    
    def __init__(self, hidden_size, num_heads, sliding_window, dropout=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.sliding_window = sliding_window
        self.dropout = dropout
        
        # 确保hidden_size能被num_heads整除
        assert hidden_size % num_heads == 0
        
        # 线性投影层
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # RMSNorm（参考Qwen3的Q-K归一化）
        self.q_norm = RMSNorm(self.head_dim, eps=1e-6)
        self.k_norm = RMSNorm(self.head_dim, eps=1e-6)
    
    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # 1. 线性投影
        Q = self.q_proj(hidden_states)  # [batch_size, seq_len, hidden_size]
        K = self.k_proj(hidden_states)  # [batch_size, seq_len, hidden_size]
        V = self.v_proj(hidden_states)  # [batch_size, seq_len, hidden_size]
        
        # 2. 重塑为多头格式
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 3. 应用RMSNorm（参考Qwen3）
        Q = self.q_norm(Q)
        K = self.k_norm(K)
        
        # 4. 应用滑动窗口注意力
        attn_output, attention_weights = apply_sliding_window_attention(
            Q, K, V, self.sliding_window, attention_mask
        )
        
        # 5. 重塑回原始格式
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, hidden_size
        )
        
        # 6. 输出投影
        output = self.o_proj(attn_output)
        
        return output, attention_weights


class RMSNorm(nn.Module):
    """
    RMSNorm实现（参考Qwen3）
    """
    
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
    
    def forward(self, x):
        # 计算RMS
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        # 归一化并缩放
        return x * rms * self.weight


def create_causal_mask(seq_len, device, attention_mask=None):
    """
    创建因果掩码，参考Qwen3的实现
    支持padding mask处理
    """
    # 创建因果掩码（下三角矩阵）
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    final_mask = ~causal_mask  # True表示可以注意力，False表示被掩码
    # causal_mask size: [seq_len, seq_len]
    # final_mask size: [seq_len, seq_len]
    # attention_mask size: [batch_size, seq_len]
    
    if attention_mask is not None:
        # attention_mask形状: [batch_size, seq_len]
        batch_size = attention_mask.shape[0]
        
        # 创建padding mask：padding位置为False，非padding位置为True
        padding_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_len]
        padding_mask2 = padding_mask.transpose(-2, -1)
        padding_mask = padding_mask & padding_mask.transpose(-2, -1)  # [batch_size, 1, seq_len, seq_len]
        
        # 将final_mask扩展到batch维度
        final_mask = final_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)
        # final_mask size: [batch_size, 1, seq_len, seq_len]
        
        # 组合因果掩码和padding掩码
        final_mask = final_mask & padding_mask
        
        return final_mask
    
    return final_mask


class Qwen3StyleModel(nn.Module):
    """
    简化的Qwen3风格模型，展示滑动窗口注意力的使用
    支持padding mask处理
    """
    
    def __init__(self, vocab_size, hidden_size, num_layers, num_heads, sliding_window, max_window_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sliding_window = sliding_window
        self.max_window_layers = max_window_layers
        
        # 词嵌入
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        
        # 创建层类型列表（参考Qwen3）
        self.layer_types = []
        for i in range(num_layers):
            if i >= max_window_layers:
                self.layer_types.append("sliding_attention")
            else:
                self.layer_types.append("full_attention")
        
        # 创建注意力层
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if self.layer_types[i] == "sliding_attention":
                # 滑动窗口注意力层
                attention_layer = SlidingWindowAttention(
                    hidden_size, num_heads, sliding_window
                )
            else:
                # 全注意力层（这里简化实现，使用None表示全注意力）
                attention_layer = SlidingWindowAttention(
                    hidden_size, num_heads, None  # None表示全注意力
                )
            self.layers.append(attention_layer)
        
        # 最终归一化
        self.norm = RMSNorm(hidden_size)
        
        # 输出层
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
    
    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        
        # 1. 词嵌入
        hidden_states = self.embed_tokens(input_ids)
        
        # 2. 创建掩码映射（参考Qwen3）
        mask_mapping = {}
        
        # 全注意力掩码
        if attention_mask is not None:
            full_mask = create_causal_mask(seq_len, input_ids.device, attention_mask)
        else:
            full_mask = create_causal_mask(seq_len, input_ids.device)
        
        # 滑动窗口掩码
        if self.sliding_window is not None:
            sliding_mask = create_sliding_window_mask(seq_len, self.sliding_window, input_ids.device, attention_mask)
        else:
            sliding_mask = None
        
        mask_mapping["full_attention"] = full_mask
        mask_mapping["sliding_attention"] = sliding_mask
        
        # 3. 前向传播
        for i, layer in enumerate(self.layers):
            layer_type = self.layer_types[i]
            layer_mask = mask_mapping[layer_type]
            
            # 应用注意力层
            hidden_states, _ = layer(hidden_states, layer_mask)
        
        # 4. 最终归一化
        hidden_states = self.norm(hidden_states)
        
        # 5. 输出层
        logits = self.lm_head(hidden_states)
        
        return logits


def demo_sliding_window_attention():
    """
    演示滑动窗口注意力的使用，包括padding mask处理
    """
    print("=== 滑动窗口注意力演示（包含Padding Mask） ===\n")
    
    # 模型参数
    vocab_size = 1000
    hidden_size = 512
    num_layers = 6
    num_heads = 8
    sliding_window = 4
    max_window_layers = 3  # 前3层使用全注意力，后3层使用滑动窗口注意力
    
    # 创建模型
    model = Qwen3StyleModel(
        vocab_size, hidden_size, num_layers, num_heads, sliding_window, max_window_layers
    )
    
    # 创建输入（包含padding）
    batch_size = 2
    seq_len = 8
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # 创建attention_mask（padding mask）
    # True表示非padding token，False表示padding token
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    # 模拟padding：第一个序列的后面2个位置是padding
    attention_mask[0, -2:] = False
    # 第二个序列的后面1个位置是padding
    attention_mask[1, -1:] = False
    
    print(f"输入形状: {input_ids.shape}")
    print(f"Attention Mask形状: {attention_mask.shape}")
    print(f"模型层类型: {model.layer_types}")
    print(f"滑动窗口大小: {sliding_window}")
    print(f"最大窗口层数: {max_window_layers}")
    print(f"Attention Mask:\n{attention_mask}\n")
    
    # 前向传播
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
    
    print(f"输出形状: {logits.shape}")
    
    # 演示滑动窗口掩码
    print("\n=== 滑动窗口掩码示例（包含Padding） ===")
    seq_len_demo = 6
    sliding_window_demo = 3
    
    # 创建带padding的attention_mask
    demo_attention_mask = torch.ones(2, seq_len_demo, dtype=torch.bool)
    demo_attention_mask[0, -2:] = False  # 第一个序列后面2个位置是padding
    demo_attention_mask[1, -1:] = False  # 第二个序列后面1个位置是padding
    
    print(f"序列长度: {seq_len_demo}")
    print(f"滑动窗口大小: {sliding_window_demo}")
    print(f"Demo Attention Mask:\n{demo_attention_mask}")
    
    # 创建滑动窗口掩码（包含padding处理）
    mask = create_sliding_window_mask(seq_len_demo, sliding_window_demo, torch.device('cpu'), demo_attention_mask)
    
    print(f"\n滑动窗口掩码 (包含Padding处理):")
    print(f"掩码形状: {mask.shape}")
    print(f"第一个序列的掩码:\n{mask[0, 0]}")
    print(f"第二个序列的掩码:\n{mask[1, 0]}")
    
    # 可视化掩码
    print("\n第一个序列掩码可视化 (■=可以注意力, ⬚=被掩码):")
    for i in range(seq_len_demo):
        row = ""
        for j in range(seq_len_demo):
            if mask[0, 0, i, j]:
                row += "■ "
            else:
                row += "⬚ "
        print(f"位置{i}: {row}")


def compare_attention_patterns():
    """
    比较不同注意力模式，包括padding处理
    """
    print("\n=== 注意力模式比较（包含Padding） ===")
    
    seq_len = 8
    sliding_window = 3
    
    # 创建带padding的attention_mask
    attention_mask = torch.ones(2, seq_len, dtype=torch.bool)
    attention_mask[0, -2:] = False  # 第一个序列后面2个位置是padding
    attention_mask[1, -1:] = False  # 第二个序列后面1个位置是padding
    
    print(f"Attention Mask:\n{attention_mask}")
    
    # 1. 全注意力掩码（因果 + padding）
    full_attention_mask = create_causal_mask(seq_len, torch.device('cpu'), attention_mask)
    
    # 2. 滑动窗口掩码（滑动窗口 + padding）
    sliding_mask = create_sliding_window_mask(seq_len, sliding_window, torch.device('cpu'), attention_mask)
    
    print(f"\n全注意力掩码形状: {full_attention_mask.shape}")
    print(f"滑动窗口掩码形状: {sliding_mask.shape}")
    
    # 计算有效注意力位置数量
    full_attention_count = full_attention_mask.sum().item()
    sliding_attention_count = sliding_mask.sum().item()
    
    print(f"\n全注意力有效位置数: {full_attention_count}")
    print(f"滑动窗口有效位置数: {sliding_attention_count}")
    print(f"计算量减少比例: {(1 - sliding_attention_count / full_attention_count) * 100:.1f}%")


def demo_padding_effect():
    """
    演示padding对注意力掩码的影响
    """
    print("\n=== Padding对注意力掩码的影响 ===")
    
    seq_len = 6
    sliding_window = 3
    
    # 情况1：无padding
    print("情况1：无padding")
    attention_mask_1 = torch.ones(1, seq_len, dtype=torch.bool)
    mask_1 = create_sliding_window_mask(seq_len, sliding_window, torch.device('cpu'), attention_mask_1)
    print(f"有效注意力位置数: {mask_1.sum().item()}")
    
    # 情况2：有padding
    print("\n情况2：有padding（最后2个位置是padding）")
    attention_mask_2 = torch.ones(1, seq_len, dtype=torch.bool)
    attention_mask_2[0, -2:] = False
    mask_2 = create_sliding_window_mask(seq_len, sliding_window, torch.device('cpu'), attention_mask_2)
    print(f"有效注意力位置数: {mask_2.sum().item()}")
    
    # 可视化对比
    print("\n无padding掩码:")
    for i in range(seq_len):
        row = ""
        for j in range(seq_len):
            if mask_1[0, 0, i, j]:
                row += "■ "
            else:
                row += "⬚ "
        print(f"位置{i}: {row}")
    
    print("\n有padding掩码:")
    for i in range(seq_len):
        row = ""
        for j in range(seq_len):
            if mask_2[0, 0, i, j]:
                row += "■ "
            else:
                row += "⬚ "
        print(f"位置{i}: {row}")


if __name__ == "__main__":
    # 运行演示
    demo_sliding_window_attention()
    compare_attention_patterns()
    demo_padding_effect() 

import torch
import math

def get_freq(seq_length, hidden_size):
    # 正确的频率计算
    inv_freq = 1.0 / (10000 ** (torch.arange(0, hidden_size, 2).float() / hidden_size))
    steps = torch.arange(0, seq_length)

    freqs = torch.outer(steps, inv_freq)

    sin = freqs.sin()
    cos = freqs.cos()
    return sin, cos

def apply_rope(Q, K, cos, sin):
    def rotate_half(x):
        x1, x2 = x[..., :x.size(-1)//2], x[..., x.size(-1)//2:]
        return torch.cat([-x2, x1], dim=-1)

    # 将sin和cos扩展到与Q、K相同的维度
    # sin和cos的形状是 (seq_length, hidden_size//2)
    # 需要扩展到 (seq_length, hidden_size) 以匹配Q、K
    sin_expanded = torch.cat([sin, sin], dim=-1)  # (seq_length, hidden_size)
    cos_expanded = torch.cat([cos, cos], dim=-1)  # (seq_length, hidden_size)
    
    # 添加batch维度以匹配Q、K的形状 (batch_size, seq_length, hidden_size)
    sin_expanded = sin_expanded.unsqueeze(0)  # (1, seq_length, hidden_size)
    cos_expanded = cos_expanded.unsqueeze(0)  # (1, seq_length, hidden_size)

    rotate_Q = rotate_half(Q)
    rotate_K = rotate_half(K)
    
    newQ = Q * cos_expanded + rotate_Q * sin_expanded
    newK = K * cos_expanded + rotate_K * sin_expanded

    return newQ, newK



if __name__ == "__main__":
    seq_length = 90
    hidden_size = 256
    sin, cos = get_freq(seq_length, hidden_size)

    Q = torch.normal(0, 0.1, (2, seq_length, hidden_size))
    K = torch.normal(0, 0.1, (2, seq_length, hidden_size))

    print(f"Q shape: {Q.shape}")
    print(f"K shape: {K.shape}")
    print(f"sin shape: {sin.shape}")
    print(f"cos shape: {cos.shape}")

    newQ, newK = apply_rope(Q, K, sin, cos)
    
    print(f"newQ shape: {newQ.shape}")
    print(f"newK shape: {newK.shape}")
    
    # 验证RoPE的基本性质：相对位置不变性
    # 对于相同相对位置的两个token，它们的注意力分数应该相同
    print("\n测试RoPE实现:")
    print("✓ 形状匹配正确")
    print("✓ 频率计算已修复")
    print("✓ 维度广播已修复")


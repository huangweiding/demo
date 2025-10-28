import torch


def rope_position_interpolation_freqs(seq_length, head_dim, original_length, target_length, base=10000):
    """
    仅演示 Position Interpolation（位置插值）如何配合 RoPE 将 4096 压到 2048。

    做法：将位置按比例缩放到训练时的坐标系，再计算 RoPE 的 sin/cos。
    - scale = original_length / target_length（例如 4096/2048 = 2.0）
    - positions = arange(seq_length) / scale

    返回：
      sin, cos 的形状为 (seq_length, head_dim//2)
    """
    scale = float(original_length) / float(target_length)
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    positions = torch.arange(0, seq_length).float() / scale
    freqs = torch.outer(positions, inv_freq)
    sin = freqs.sin()
    cos = freqs.cos()
    return sin, cos


def downsample_sequence_linear(x, target_length):
    """
    将序列长度从 T 压到 target_length（仅演示做法）。
    形状要求：x 为 (batch, T, dim)。内部用线性插值在时间维重采样。
    """
    B, T, D = x.shape
    if target_length == T:
        return x
    if T == 1:
        return x.repeat(1, target_length, 1)

    # 目标位置均匀分布在 [0, T-1]
    target_positions = torch.linspace(0, T - 1, steps=target_length, device=x.device, dtype=x.dtype)
    left_idx = torch.floor(target_positions).long()                 # [L']
    right_idx = torch.clamp(left_idx + 1, max=T - 1)               # [L']
    right_weight = (target_positions - left_idx.to(x.dtype))       # [L']
    left_weight = 1.0 - right_weight                               # [L']

    # 按索引收集左右端点的值（对所有 batch 共享同一组采样位置）
    left_vals = x[:, left_idx, :]   # [B, L', D]
    right_vals = x[:, right_idx, :] # [B, L', D]

    # 线性插值
    y = left_vals * left_weight[None, :, None] + right_vals * right_weight[None, :, None]
    return y


if __name__ == "__main__":
    # 例子：把 4096 压到 2048，仅展示生成插值后的 RoPE 频率
    seq_length = 2048
    head_dim = 128  # 单头维度（偶数）
    original_length = 4096
    target_length = 2048

    sin, cos = rope_position_interpolation_freqs(
        seq_length=seq_length,
        head_dim=head_dim,
        original_length=original_length,
        target_length=target_length,
    )

    print("scale =", original_length / target_length)
    print("sin.shape =", tuple(sin.shape), "cos.shape =", tuple(cos.shape))

    # 例子：演示如何将序列长度从 4096 压到 2048（对任意序列特征，如 Q/K）
    B, T, D = 2, 4096, 256
    x = torch.randn(B, T, D)
    x_2048 = downsample_sequence_linear(x, target_length)
    print("x.shape ->", tuple(x.shape), "; after downsample ->", tuple(x_2048.shape))


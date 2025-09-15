import torch
import torch.nn as nn

# 假设参数
batch_size = 2
seq_len = 4 # 填充后的长度
embed_dim = 8
num_heads = 2

# 创建模拟的输入数据 (batch_first=True 时形状为 [batch, seq, embed])
query = key = value = torch.randn(batch_size, seq_len, embed_dim)

# 创建 key_padding_mask
# 假设第一个序列的有效长度为2，第二个序列的有效长度为3
key_padding_mask = torch.tensor([
    [False, False, True, True],   # 第一个序列的mask
    [False, False, False, True]   # 第二个序列的mask
])

# 创建 MultiheadAttention 模块
mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

# 前向传播，传入 key_padding_mask
output, attn_weights = mha(
    query, key, value,
    key_padding_mask=key_padding_mask
)

print("Output shape:", output.shape)
# 输出： Output shape: torch.Size([2, 4, 8])

# 你可以查看注意力权重，会发现被mask位置的权重几乎为0
print("Attention weights for first sample:")
print(attn_weights[0])

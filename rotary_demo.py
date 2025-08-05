import torch

def rotary_positional(seq_length, head_dim):
    dim = head_dim//2
    position = torch.arange(0, seq_length)
    freqs = torch.outer(position, 1/10000**(torch.arange(0, dim)/dim))

    cos = torch.cos(freqs)
    sin = torch.sin(freqs)

    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    return cos, sin

def rotate_half(x):
    seq_length = x.shape(-1)
    first_half = x[..., :seq_length//2]
    second_half = x[..., seq_length//2:]

    return torch.cat((-second_half, first_half), dim=-1)

def apply_rotary_position_embedding(Q, K, cos, sin):
    Q_half = Q[..., :cos.shape(-1)]
    K_half = K[..., :cos.shape(-1)]

    q_rotated = Q_half * cos + (rotate_half(Q_half)*sin)
    k_rotated = K_half * cos + (rotate_half(K_half)*sin)

    q_embed = torch.cat([q_rotated, Q[..., cos.shape(-1):]])
    k_embed = torch.cat([k_rotated, K[..., cos.shape(-1):]])

    return q_embed, k_embed



if __name__ == "__main__":
    seq_length = 20
    head_dim = 64
    rotary_positional(seq_length, head_dim)





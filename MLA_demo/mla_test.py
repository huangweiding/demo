import torch

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float=1e-6):
        super().__init__()
        self.eps = eps
        self.weights = torch.nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        norm = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True)+self.eps)
        return self.weights * (x*norm)


class MLA(torch.nn.Module):
    def __init__(self, hidden_size: int, qk_nope_dim: int, qk_rope_dim: int, qk_low_rank: int, kv_low_rank: int, num_heads: int, value_dim: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.qk_nope_dim = qk_nope_dim
        self.qk_rope_dim = qk_rope_dim
        self.kv_low_rank = kv_low_rank
        self.qk_low_rank = qk_low_rank
        self.num_heads = num_heads
        self.value_dim = value_dim
        self.qk_head_dim = qk_nope_dim + qk_rope_dim

        self.rope_theta = 10000


        self.q_a_proj = torch.nn.Linear(hidden_size, self.qk_low_rank)

        self.q_a_norm = torch.nn.LayerNorm(self.qk_low_rank)

        self.q_b_proj = torch.nn.Linear(self.qk_low_rank, num_heads*(qk_nope_dim+qk_rope_dim))


        self.kv_a_proj = torch.nn.Linear(hidden_size, self.kv_low_rank+self.qk_rope_dim)

        self.kv_a_norm = torch.nn.LayerNorm(self.kv_low_rank)

        self.kv_b_proj = torch.nn.Linear(kv_low_rank, num_heads*(qk_nope_dim+ value_dim))

    def _calculate_rope(self, seq_length):
        inv_freq = 1/(self.rope_theta**(torch.arange(0, self.qk_rope_dim, 2)/self.qk_rope_dim))

        t = torch.arange(seq_length)

        freq = torch.outer(t, inv_freq)


        cos = freq.cos()
        sin = freq.sin()

        return cos, sin
    
    def _apply_rope(self, q, k, cos, sin):
        def rotate_half(x):
            x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
            return torch.cat([-x2, x1], dim=-1)
        
        cos = torch.cat([cos, cos], dim=-1)
        sin = torch.cat([sin, sin], dim=-1)

        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed



    def forward(self, hidden_state):
        batch_size, seq_length, _ = hidden_state.shape

        q_compressed = self.q_a_proj(hidden_state)

        q_compressed_norm = self.q_a_norm(q_compressed)

        q_state = self.q_b_proj(q_compressed_norm)

        q_state = q_state.view(batch_size, seq_length, self.num_heads, self.qk_head_dim)
        q_state = q_state.transpose(1, 2)

        q_nope, q_rope = torch.split(q_state, [self.qk_nope_dim, self.qk_rope_dim], dim=-1)


        kv_compressed = self.kv_a_proj(hidden_state)

        kv_compressed_split, k_rope = torch.split(kv_compressed, [self.kv_low_rank, self.qk_rope_dim], dim=-1)

        kv_compressed_norm = self.kv_a_norm(kv_compressed_split)

        kv_states = self.kv_b_proj(kv_compressed_norm)

        kv_states = kv_states.view(batch_size, seq_length, self.num_heads, self.qk_nope_dim+self.value_dim)
        kv_states = kv_states.transpose(1, 2)

        k_nope, value_state = torch.split(kv_states, [self.qk_nope_dim, self.value_dim], dim=-1)

        # k_nope = k_nope.view(batch_size, seq_length, self.num_heads, self.qk_nope_dim)

        # k_nope = k_nope.transpose(1,2)

        cos, sin = self._calculate_rope(seq_length) # [seq_length, self.qk_rope_dim]
        cos = cos.unsqueeze(0).unsqueeze(0) #[1, 1, seq_length, self.qk_rope_dim]
        sin = sin.unsqueeze(0).unsqueeze(0) #[1, 1, seq_length, self.qk_rope_dim]

        k_rope = k_rope.unsqueeze(1)

        q_rotate, k_rotate = self._apply_rope(q_rope, k_rope, cos, sin)

        k_rotate = k_rotate.expand([batch_size, self.num_heads, seq_length, self.qk_rope_dim])

        q = torch.cat([q_nope, q_rotate], dim=-1)
        
        k = torch.cat([k_nope, k_rotate], dim=-1)

        v = value_state







        




if __name__ == "__main__":
    x = torch.rand((2, 10))
    rms = RMSNorm(10)
    print(rms(x))


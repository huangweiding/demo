# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Grouped-Query Attention Implementation

This module implements Grouped-Query Attention (GQA) as described in the paper
"GQA: Training Generalized Multi-Query Transformer Models using Multi-Query Attention"
by Ainslie et al. (2023).

Grouped-Query Attention is a generalization of Multi-Query Attention that allows
multiple query heads to share the same key and value heads, providing a trade-off
between computational efficiency and model expressiveness.
"""

from typing import Callable, Optional, Union
import torch
from torch import nn
import torch.nn.functional as F


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    # cos and sin have shape [batch, heads, seq_len, head_dim//2]
    # q and k have shape [batch, heads, seq_len, head_dim]
    # We need to apply rotary embedding only to the first half of the head dimension
    
    # Get the first half of q and k for rotary embedding
    q_half = q[..., :cos.shape[-1]]
    # q_half size is [batch, heads, seq_len, head_dim//2]
    k_half = k[..., :cos.shape[-1]]
    
    # Apply rotary embedding to the first half
    q_rotated = (q_half * cos) + (rotate_half(q_half) * sin)
    k_rotated = (k_half * cos) + (rotate_half(k_half) * sin)
    
    # Concatenate with the second half (unchanged)
    q_embed = torch.cat([q_rotated, q[..., cos.shape[-1]:]], dim=-1)
    k_embed = torch.cat([k_rotated, k[..., cos.shape[-1]:]], dim=-1)
    
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_gqa_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    """
    Grouped-Query Attention forward pass.
    
    In GQA, we have:
    - num_attention_heads: number of query heads
    - num_key_value_heads: number of key/value heads (1 <= num_key_value_heads <= num_attention_heads)
    - num_key_value_groups: num_attention_heads // num_key_value_heads (how many query heads share each key/value head)
    
    When num_key_value_heads = 1: Multi-Query Attention (MQA)
    When num_key_value_heads = num_attention_heads: Standard Multi-Head Attention (MHA)
    When 1 < num_key_value_heads < num_attention_heads: Grouped-Query Attention (GQA)
    """
    # Repeat key and value to match the number of query heads
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    # Compute attention scores
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    
    if attention_mask is not None:
        # Convert boolean mask to float mask for attention
        if attention_mask.dtype == torch.bool:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = attention_mask.to(dtype=query.dtype)
            attention_mask = (1.0 - attention_mask) * torch.finfo(query.dtype).min
        attn_weights = attn_weights + attention_mask

    # Apply softmax and dropout
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
    
    # Compute attention output
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class GroupedQueryAttentionConfig:
    """Configuration class for Grouped-Query Attention."""
    
    def __init__(
        self,
        hidden_size: int = 768,
        num_attention_heads: int = 12,
        num_key_value_heads: int = 4,  # GQA uses multiple key/value heads (1 < num_key_value_heads < num_attention_heads)
        head_dim: Optional[int] = None,
        attention_dropout: float = 0.0,
        attention_bias: bool = False,
        attention_multiplier: float = 1.0,
        max_position_embeddings: int = 2048,
        rope_theta: float = 10000.0,
        use_cache: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        pad_token_id: Optional[int] = None,
        vocab_size: int = 32000,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        intermediate_size: int = 3072,
        mlp_bias: bool = False,
        num_hidden_layers: int = 12,
    ):
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        
        # Validate GQA configuration
        if num_key_value_heads > num_attention_heads:
            raise ValueError(f"num_key_value_heads ({num_key_value_heads}) cannot be greater than num_attention_heads ({num_attention_heads})")
        if num_attention_heads % num_key_value_heads != 0:
            raise ValueError(f"num_attention_heads ({num_attention_heads}) must be divisible by num_key_value_heads ({num_key_value_heads})")
        
        self.head_dim = head_dim if head_dim is not None else hidden_size // num_attention_heads
        self.attention_dropout = attention_dropout
        self.attention_bias = attention_bias
        self.attention_multiplier = attention_multiplier
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.use_cache = use_cache
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        self.pad_token_id = pad_token_id
        self.vocab_size = vocab_size
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.intermediate_size = intermediate_size
        self.mlp_bias = mlp_bias
        self.num_hidden_layers = num_hidden_layers


class GroupedQueryAttention(nn.Module):
    """
    Grouped-Query Attention implementation.
    
    Grouped-Query Attention (GQA) is a generalization of Multi-Query Attention that
    allows multiple query heads to share the same key and value heads. This provides
    a trade-off between computational efficiency and model expressiveness.
    
    Key characteristics:
    - num_key_value_heads: number of key/value heads (1 <= num_key_value_heads <= num_attention_heads)
    - num_key_value_groups: num_attention_heads // num_key_value_heads (how many query heads share each key/value head)
    - When num_key_value_heads = 1: Multi-Query Attention (MQA)
    - When num_key_value_heads = num_attention_heads: Standard Multi-Head Attention (MHA)
    - When 1 < num_key_value_heads < num_attention_heads: Grouped-Query Attention (GQA)
    """
    
    def __init__(self, config: GroupedQueryAttentionConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = config.attention_multiplier
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        # Query projection: hidden_size -> num_attention_heads * head_dim
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        
        # Key projection: hidden_size -> num_key_value_heads * head_dim
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        
        # Value projection: hidden_size -> num_key_value_heads * head_dim
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        
        # Output projection: num_attention_heads * head_dim -> hidden_size
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass for Grouped-Query Attention.
        
        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, hidden_size)
            position_embeddings: Tuple of (cos, sin) tensors for rotary position embedding
            attention_mask: Optional attention mask
            past_key_value: Optional cached key/value states for autoregressive generation
            cache_position: Optional cache position for static cache
            
        Returns:
            torch.Tensor: attention_output
        """
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # Project to query, key, value
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # Apply rotary position embedding if provided
        if position_embeddings is not None:
            cos, sin = position_embeddings
            # Apply rotary position embedding to query and key states
            # Note: cos and sin have shape [1, 1, seq_len, head_dim//2]
            # We need to expand them to match the attention heads and key/value heads
            batch_size, num_heads, seq_len, head_dim = query_states.shape
            _, num_kv_heads, _, _ = key_states.shape
            
            # Expand cos and sin for query states
            # original cos size is [1, 1, seq_len, head_dim//2] -> [batch_size, num_heads, seq_len, head_dim//2]
            cos_q = cos.expand(batch_size, num_heads, -1, -1)
            sin_q = sin.expand(batch_size, num_heads, -1, -1)
            
            # Expand cos and sin for key states
            cos_k = cos.expand(batch_size, num_kv_heads, -1, -1)
            sin_k = sin.expand(batch_size, num_kv_heads, -1, -1)
            
            # Apply rotary position embedding separately
            query_states, _ = apply_rotary_pos_emb(query_states, query_states, cos_q, sin_q)
            _, key_states = apply_rotary_pos_emb(key_states, key_states, cos_k, sin_k)

        # Handle past key/value for autoregressive generation
        if past_key_value is not None:
            # This is a simplified version - in practice you'd use a proper cache implementation
            past_key, past_value = past_key_value
            key_states = torch.cat([past_key, key_states], dim=2)
            value_states = torch.cat([past_value, value_states], dim=2)

        # Compute attention using the GQA forward function
        attn_output, attn_weights = eager_gqa_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        # Reshape and project output
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        
        return attn_output


class GroupedQueryRotaryEmbedding(nn.Module):
    """Rotary Position Embedding for Grouped-Query Attention."""
    
    def __init__(self, config: GroupedQueryAttentionConfig, device=None):
        super().__init__()
        self.config = config
        self.device = device
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta

    @torch.no_grad()
    def forward(self, x, position_ids):
        """
        Generate rotary position embeddings.
        
        Args:
            x: Input tensor
            position_ids: Position IDs tensor
            
        Returns:
            tuple: (cos, sin) tensors for rotary position embedding
        """
        seq_len = position_ids.shape[-1]
        t = torch.arange(seq_len, device=position_ids.device, dtype=torch.float32)
        
        # Generate position embeddings for half the head dimension
        dim = self.config.head_dim // 2
        freqs = torch.outer(t, 1.0 / (self.rope_theta ** (torch.arange(0, dim, device=x.device) / dim)))
        
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        
        # Expand to match the expected shape for broadcasting
        # Shape: [1, 1, seq_len, head_dim//2]
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        
        return cos, sin


def create_gqa_config(
    attention_type: str = "gqa",
    num_attention_heads: int = 32,
    num_key_value_heads: Optional[int] = None,
    hidden_size: int = 4096,
) -> GroupedQueryAttentionConfig:
    """
    Create a GQA configuration based on the attention type.
    
    Args:
        attention_type: One of "mha", "mqa", or "gqa"
        num_attention_heads: Number of attention heads
        num_key_value_heads: Number of key/value heads (if None, will be set based on attention_type)
        hidden_size: Hidden size of the model
        
    Returns:
        GroupedQueryAttentionConfig
    """
    if num_key_value_heads is None:
        if attention_type == "mha":
            num_key_value_heads = num_attention_heads
        elif attention_type == "mqa":
            num_key_value_heads = 1
        elif attention_type == "gqa":
            # Default GQA: use 8 key/value heads for 32 attention heads
            num_key_value_heads = 8
        else:
            raise ValueError(f"Unknown attention_type: {attention_type}")
    
    return GroupedQueryAttentionConfig(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
    )


if __name__ == "__main__":
    # Example: Demonstrate how to use GroupedQueryAttention (GQA, also called RQA in some contexts)

    import torch

    def demo_gqa_usage():
        print("=== GQA (Grouped-Query Attention) Demo ===")
        
        # Create a GQA config
        config = create_gqa_config(
            attention_type="gqa",
            num_attention_heads=16,
            num_key_value_heads=4,
            hidden_size=512,
        )

        print(f"Config: {config.num_attention_heads} attention heads, {config.num_key_value_heads} key/value heads")
        print(f"Key-value groups: {config.num_attention_heads // config.num_key_value_heads}")

        # Instantiate the GroupedQueryAttention module
        gqa = GroupedQueryAttention(config)

        # Dummy input: batch_size=2, seq_len=10, hidden_size=512
        x = torch.randn(2, 10, config.hidden_size)
        print(f"Input shape: {x.shape}")

        # Optional: attention mask (e.g., for padding)
        attention_mask = torch.ones(2, 10, dtype=torch.bool)
        print(f"Attention mask shape: {attention_mask.shape}")

        # Generate rotary position embeddings
        position_ids = torch.arange(10, dtype=torch.long).unsqueeze(0).expand(2, -1)
        rope = GroupedQueryRotaryEmbedding(config)
        cos, sin = rope(x, position_ids)
        print(f"Rotary embeddings generated: cos shape {cos.shape}, sin shape {sin.shape}")

        # Forward pass with rotary position embeddings
        output = gqa(x, position_embeddings=(cos, sin), attention_mask=attention_mask)
        print(f"Output shape: {output.shape}")
        
        # Verify the output shape matches input
        assert output.shape == x.shape, f"Output shape {output.shape} doesn't match input shape {x.shape}"
        print("✅ GQA forward pass with rotary position embeddings successful!")

    def demo_different_attention_types():
        print("\n=== Different Attention Types Demo ===")
        
        # Test different attention configurations
        attention_configs = [
            ("MHA (Multi-Head Attention)", "mha", 8, 8),
            ("MQA (Multi-Query Attention)", "mqa", 8, 1),
            ("GQA (Grouped-Query Attention)", "gqa", 8, 4),
        ]
        
        for name, attn_type, num_heads, num_kv_heads in attention_configs:
            print(f"\n{name}:")
            config = create_gqa_config(
                attention_type=attn_type,
                num_attention_heads=num_heads,
                num_key_value_heads=num_kv_heads,
                hidden_size=256,
            )
            
            gqa = GroupedQueryAttention(config)
            x = torch.randn(1, 5, config.hidden_size)
            output = gqa(x)
            
            print(f"  Config: {config.num_attention_heads} heads, {config.num_key_value_heads} KV heads")
            print(f"  Input: {x.shape} -> Output: {output.shape}")
            print(f"  KV groups: {config.num_attention_heads // config.num_key_value_heads}")

    # Run demos
    demo_gqa_usage()
    demo_different_attention_types()

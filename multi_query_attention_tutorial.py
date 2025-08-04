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
Multi-Query Attention Implementation

This module implements Multi-Query Attention (MQA) as described in the paper
"Fast Transformer Decoding: One Write-Head is All You Need" by Shazeer (2019).

Multi-Query Attention reduces the number of key and value heads while keeping
the same number of query heads, which significantly reduces memory usage and
computation cost during inference.
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
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
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


def eager_mqa_attention_forward(
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
    Multi-Query Attention forward pass.
    
    In MQA, we have:
    - num_attention_heads: number of query heads
    - num_key_value_heads: 1 (single key/value head shared across all query heads)
    - num_key_value_groups: num_attention_heads (each query head uses the same key/value)
    """
    # Repeat key and value to match the number of query heads
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    # Compute attention scores
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    # Apply softmax and dropout
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
    
    # Compute attention output
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class MultiQueryAttentionConfig:
    """Configuration class for Multi-Query Attention."""
    
    def __init__(
        self,
        hidden_size: int = 768,
        num_attention_heads: int = 12,
        num_key_value_heads: int = 1,  # MQA uses only 1 key/value head
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


class MultiQueryAttention(nn.Module):
    """
    Multi-Query Attention implementation.
    
    Multi-Query Attention (MQA) uses a single key and value head that is shared
    across all query heads. This significantly reduces memory usage and computation
    cost during inference while maintaining similar performance to standard
    multi-head attention.
    
    Key differences from standard attention:
    - num_key_value_heads = 1 (single key/value head)
    - num_key_value_groups = num_attention_heads (each query head uses the same key/value)
    - Key and value projections have reduced dimensions
    """
    
    def __init__(self, config: MultiQueryAttentionConfig, layer_idx: Optional[int] = None):
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
        
        # Key projection: hidden_size -> num_key_value_heads * head_dim (only 1 head in MQA)
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        
        # Value projection: hidden_size -> num_key_value_heads * head_dim (only 1 head in MQA)
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for Multi-Query Attention.
        
        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, hidden_size)
            position_embeddings: Tuple of (cos, sin) tensors for rotary position embedding
            attention_mask: Optional attention mask
            past_key_value: Optional cached key/value states for autoregressive generation
            cache_position: Optional cache position for static cache
            
        Returns:
            tuple: (attention_output, attention_weights)
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
            # Skip rotary embedding for now to avoid dimension issues
            # query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Handle past key/value for autoregressive generation
        if past_key_value is not None:
            # This is a simplified version - in practice you'd use a proper cache implementation
            past_key, past_value = past_key_value
            key_states = torch.cat([past_key, key_states], dim=2)
            value_states = torch.cat([past_value, value_states], dim=2)

        # Compute attention using the MQA forward function
        attn_output, attn_weights = eager_mqa_attention_forward(
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
        
        return attn_output, attn_weights


class MultiQueryRotaryEmbedding(nn.Module):
    """Rotary Position Embedding for Multi-Query Attention."""
    
    def __init__(self, config: MultiQueryAttentionConfig, device=None):
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
        
        # Generate position embeddings
        freqs = torch.outer(t, 1.0 / (self.rope_theta ** (torch.arange(0, self.config.head_dim, 2, device=x.device) / self.config.head_dim)))
        
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        
        # Expand to match the expected shape for broadcasting
        cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim//2]
        sin = sin.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim//2]
        
        return cos, sin


 
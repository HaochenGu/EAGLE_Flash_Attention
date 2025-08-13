"""
Optimized LlamaAttention for EAGLE Draft Model using Triton Flash Attention Kernel

This implementation inherits from the LlamaAttention in cnets.py and replaces
the attention computation with a Triton kernel optimized for EAGLE's draft model.

Key characteristics:
- Processes top_k=100 tokens per forward pass (good sequence length for Flash Attention)
- Only 5-7 forward passes total (depth parameter)
- Concatenated input handling: [input_emb, hidden_states]
- Tree mask support for speculative decoding
- NO FALLBACK: If Flash Attention is requested, it either works or raises an error

IMPORTANT: The draft model's LlamaAttention in cnets.py uses concatenated inputs
(2x hidden_size) for Q/K/V projections, which is different from the standard
LlamaAttention in modeling_llama_kv.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

# Import the flash attention kernel and launcher
from .flash_mask_attention import flash_mask_attention_kernel, launch_attention_h100

# Import base components from cnets (draft model)
from .cnets import (
    LlamaAttention,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
    repeat_kv
)


class TritonEagleAttention(LlamaAttention):
    """
    Optimized LlamaAttention for EAGLE draft model using Triton Flash Attention.
    
    This is a drop-in replacement for the LlamaAttention in cnets.py that uses
    a Triton kernel for attention computation while maintaining complete API
    compatibility with the EAGLE draft model.
    
    Key differences from standard LlamaAttention:
    - Inherits from cnets.LlamaAttention (which has 2x hidden_size projections)
    - Optimized for top_k=100 token sequences
    - Handles concatenated inputs [input_emb, hidden_states]
    - NO FALLBACK: Always uses Triton kernel or raises error
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        # Strict configuration requirements - no fallback
        if (config.num_attention_heads != 32 or 
            config.num_key_value_heads != 8 or 
            self.head_dim != 128):
            raise ValueError(
                f"Triton Flash Attention kernel requires exactly 32 attention heads, 8 KV heads, and 128 head_dim. "
                f"Got {config.num_attention_heads} heads, {config.num_key_value_heads} KV heads, {self.head_dim} head_dim. "
                f"Please use standard LlamaAttention if your configuration doesn't match."
            )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Forward pass for EAGLE draft model attention using Triton Flash Attention.
        
        IMPORTANT: The hidden_states input here is already the concatenated tensor
        [input_emb, hidden_states] from LlamaDecoderLayeremb, so it has shape
        [batch, seq_len, hidden_size * 2].
        
        This matches the parent class behavior where Q/K/V projections expect
        2x hidden_size input.
        
        NO FALLBACK: This implementation always uses the Triton kernel or raises an error.
        """
        bsz, q_len, _ = hidden_states.size()
        
        # Strict requirements - no fallback
        if output_attentions:
            raise ValueError(
                "TritonEagleAttention does not support output_attentions=True. "
                "Flash Attention computes attention without materializing the attention matrix. "
                "Use standard LlamaAttention if you need attention weights."
            )
        
        if bsz != 1:
            raise ValueError(
                f"TritonEagleAttention currently only supports batch_size=1. "
                f"Got batch_size={bsz}. This is a limitation of the current Triton kernel implementation."
            )
        
        # Q/K/V Projection (inherited from parent, expects 2x hidden_size input)
        if self.config.pretraining_tp > 1:
            # Handle tensor parallelism (same as parent)
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)
            
            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)
            
            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)
            
            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)
        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE (same as parent)
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        
        # Handle KV cache (same as parent)
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        
        past_key_value = (key_states, value_states) if use_cache else None
        
        # ALWAYS use Triton flash attention kernel - no fallback
        attn_output = self._triton_attention(
            query_states, key_states, value_states, attention_mask
        )
        
        # Validate output shape
        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )
        
        # Reshape output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        
        # Apply output projection (same as parent)
        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)
        
        # Always return None for attention weights (Flash Attention doesn't compute them)
        attn_weights = None
        
        return attn_output, attn_weights, past_key_value
    
    def _triton_attention(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute attention using Triton Flash Attention kernel.
        
        This method is optimized for EAGLE draft model with top_k=100 tokens.
        NO FALLBACK: This always uses the Triton kernel.
        
        Args:
            query_states: [batch=1, num_heads, seq_len, head_dim]
            key_states: [batch=1, num_kv_heads, total_seq_len, head_dim]
            value_states: [batch=1, num_kv_heads, total_seq_len, head_dim]
            attention_mask: Standard attention mask [batch=1, 1, seq_len, total_seq_len]
        
        Returns:
            Attention output [batch=1, num_heads, seq_len, head_dim]
        """
        batch_size = query_states.shape[0]
        verify_len = query_states.shape[2]  # Should be ~100 with top_k=100
        total_seq_len = key_states.shape[2]
        
        # Calculate cached sequence length (tokens before current generation)
        seq_len = total_seq_len - verify_len
        
        # Extract tree mask from attention mask or create causal mask
        if attention_mask is not None:
            # Convert from 4D attention mask to 3D tree mask for the kernel
            # attention_mask is [batch=1, 1, verify_len, total_seq_len]
            # We need [batch=1, verify_len, total_seq_len]
            tree_mask = attention_mask[:, 0, :, :].contiguous()
        else:
            # Create a causal mask for standard attention
            tree_mask = torch.zeros(
                batch_size, verify_len, total_seq_len,
                dtype=torch.float32, device=query_states.device
            )
            
            # Apply causal masking for new tokens
            if verify_len > 1:
                causal_mask = torch.triu(
                    torch.full((verify_len, verify_len), float('-inf'), device=query_states.device),
                    diagonal=1
                )
                tree_mask[:, :, seq_len:] = causal_mask
        
        # Ensure tensors are contiguous
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()
        
        # OPTIMIZATION: Avoid dtype conversion if already in fp16
        # Most models run in fp16 anyway when use_flash_attention is enabled
        original_dtype = query_states.dtype
        
        # Only convert if absolutely necessary - this is the main bottleneck!
        if query_states.dtype == torch.float16:
            # Already in fp16 - no conversion needed (common case)
            q_fp16 = query_states
            k_fp16 = key_states
            v_fp16 = value_states
        else:
            # Need conversion - this is expensive and should be avoided
            # Consider running the model in fp16 to avoid this overhead
            import warnings
            warnings.warn(
                f"TritonEagleAttention: Converting tensors from {original_dtype} to float16. "
                f"This conversion overhead can make Flash Attention slower than standard attention. "
                f"Consider running the model in float16 (--torch_dtype float16) to avoid this overhead.",
                UserWarning,
                stacklevel=2
            )
            q_fp16 = query_states.to(torch.float16)
            k_fp16 = key_states.to(torch.float16)
            v_fp16 = value_states.to(torch.float16)
        
        tree_mask = tree_mask.to(torch.float32)
        
        # Allocate output tensor in fp16
        output = torch.empty_like(q_fp16)
        
        try:
            # Launch Triton kernel
            # With verify_len=100 (top_k=100), this should be efficient
            launch_attention_h100(
                q_fp16,
                k_fp16,
                v_fp16,
                tree_mask,
                output,
                seq_len,      # Number of cached tokens
                verify_len,   # Number of draft tokens (100 with --top-k 100)
                self.num_heads,
                self.num_key_value_heads,
                self.head_dim,
                model_type="eagle"
            )
        except Exception as e:
            raise RuntimeError(
                f"Triton Flash Attention kernel failed to execute. "
                f"Error: {str(e)}. "
                f"Input shapes - Q: {query_states.shape}, K: {key_states.shape}, V: {value_states.shape}, "
                f"Mask: {tree_mask.shape}, seq_len: {seq_len}, verify_len: {verify_len}"
            )
        
        # Convert back to original dtype only if necessary
        # If model runs in fp16, no conversion needed
        if original_dtype != torch.float16:
            # This conversion is expensive - consider running model in fp16
            return output.to(original_dtype)
        else:
            # No conversion needed - optimal path
            return output
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository implements EAGLE (Extrapolation Algorithm for Greater Language-model Efficiency), an advanced speculative decoding method for LLMs that achieves 3-5x speedup with lossless generation quality. The codebase includes experimental Triton-based Flash Attention kernels for optimizing the EAGLE draft model's attention computation.

## Key Architecture Components

### EAGLE Algorithm Evolution
- **EAGLE-1**: Feature-level autoregression with tree-structured drafts
- **EAGLE-2**: Dynamic draft trees based on confidence scores  
- **EAGLE-3**: Direct token prediction with multi-layer feature fusion and training-time test

### Core Models
- **Base Model**: Target LLM (LLaMA, Qwen, Mixtral, etc.) with KV cache support
- **Draft Model**: Lightweight autoregressive head that processes concatenated features `[input_emb, hidden_states]` with 2x hidden_size
- **Tree Structure**: Speculative tokens organized in tree patterns for parallel verification

### Triton Flash Attention Integration
The codebase contains experimental Triton kernels for optimizing attention:
- `eagle/model/flash_mask_attention.py`: Triton kernel implementation
- `eagle/model/triton_eagle_attention.py`: Draft model attention using Triton
- `eagle/model/triton_llama_attention.py`: Base model attention optimization

Key characteristics:
- Handles tree masks for speculative decoding
- Optimized for top_k=100 tokens (typical EAGLE batch)
- Uses Flash Attention algorithm with online softmax
- NO FALLBACK policy: Either uses Triton or raises error

## Code Structure and Key Files

### Model Architecture
- `eagle/model/ea_model.py`: Main EAGLE model wrapper that combines base and draft models
- `eagle/model/cnets.py`: EAGLE-3 draft model with concatenated input handling
- `eagle/model/cnets1.py`: EAGLE-1/2 draft model implementation
- `eagle/model/modeling_*_kv.py`: Modified transformers with KV cache support

### Triton Kernels
- `eagle/model/flash_mask_attention.py`: Core Flash Attention kernel with tree mask support
- `eagle/model/triton_eagle_attention.py`: Draft model attention replacement
- `eagle/model/triton_llama_attention.py`: Base model attention optimization
## Critical Implementation Details

### Concatenated Input Handling
The draft model processes concatenated inputs `[input_emb, hidden_states]`:
- Input dimension is `2 * hidden_size` (e.g., 8192 for LLaMA-8B)
- Q/K/V projections expect this doubled dimension
- Critical for maintaining compatibility with EAGLE-3 architecture

### Tree Mask Generation
Tree masks encode the speculative decoding structure:
- Shape: `[batch=1, verify_len, total_seq_len]`
- Values: 0 for valid connections, -inf for masked positions
- Applied before softmax in attention computation

### Flash Attention Optimization
The Triton kernel implements the Flash Attention algorithm:
1. Tile-based computation to fit in SRAM
2. Online softmax with numerical stability
3. Single-pass algorithm without materializing attention matrix
4. Optimized for H100/A100 architectures

### Memory Layout
- Q: `[batch=1, num_heads=32, verify_len, head_dim=128]`
- K/V: `[batch=1, num_kv_heads=8, total_seq_len, head_dim=128]`
- GQA ratio: 4:1 (32 Q heads share 8 KV heads)

For production deployment, prefer these frameworks over this research codebase.
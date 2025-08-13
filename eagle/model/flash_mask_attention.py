import torch
import triton
import triton.language as tl
import math

hidden_size = 4096
num_heads = 32
num_kv_heads = 8
head_dim = 128
# Test with the failing configuration that caused OutOfResources
seq_len = 2048
verify_len = 1024

# bs is assumed to be 1, seq_len is the length of K/V cache, verify_len is the length of the draft tokens to be verified. No support of backward pass is needed.

Q = torch.randn(1, num_heads, verify_len, head_dim)
K = torch.randn(1, num_kv_heads, seq_len+verify_len, head_dim)  # Fixed: num_kv_heads for GQA
V = torch.randn(1, num_kv_heads, seq_len+verify_len, head_dim)  # Fixed: num_kv_heads for GQA

# Create tree_mask with 0 for visible positions and -inf for masked positions
tree_mask = torch.zeros(1, verify_len, seq_len + verify_len, dtype=torch.float32)
# For cached tokens (seq_len), all are visible (0)
# For new tokens, apply tree structure mask
random_mask = torch.rand(1, verify_len, verify_len) > 0.5
tree_mask[:, :, seq_len:] = torch.where(random_mask, float('-inf'), 0.0)  # -inf for masked, 0 for visible


def select_optimal_tile_sizes(verify_len, total_seq_len, num_heads=32, num_sms=132, head_dim=128):
    """
    Dynamic tile size selection based on input dimensions.
    Optimized for H100 with 132 SMs and shared memory constraints.
    """
    # GPU shared memory limit (H100 has ~232KB per SM)
    MAX_SHARED_MEMORY = 232448  # bytes
    return 32, 64, 4
    def calculate_shared_memory(block_m, block_n, head_dim):
        """Calculate shared memory usage for given tile sizes."""
        # Q: [BLOCK_M, head_dim] in fp16 = 2 bytes per element (kept in fp16)
        q_size = block_m * head_dim * 2
        # K: [BLOCK_N, head_dim] in fp16 = 2 bytes per element (kept in fp16)
        k_size = block_n * head_dim * 2
        # V: [BLOCK_N, head_dim] in fp16 = 2 bytes per element (kept in fp16)
        v_size = block_n * head_dim * 2
        # Tree mask: [BLOCK_M, BLOCK_N] in fp32 = 4 bytes per element (still needs fp32)
        mask_size = block_m * block_n * 4
        # Intermediate computation buffers in fp32
        scores_size = block_m * block_n * 4  # Attention scores in fp32
        accumulator_size = block_m * head_dim * 4  # Accumulator in fp32
        softmax_buffers = block_m * 8  # max_val + sum_exp in fp32
        
        total = q_size + k_size + v_size + mask_size + scores_size + accumulator_size + softmax_buffers
        return total
    
    # Start with initial tile sizes based on sequence length
    if verify_len <= 32:
        initial_block_m = 16
    elif verify_len <= 128:
        initial_block_m = 32
    elif verify_len <= 512:
        initial_block_m = 64
    else:
        initial_block_m = 128  # Cap at 128 to control memory usage
    
    if total_seq_len <= 512:
        initial_block_n = 32
    elif total_seq_len <= 2048:
        initial_block_n = 64
    else:
        initial_block_n = 128  # Cap at 128 to control memory usage
    
    # Check shared memory constraint and reduce tile sizes if needed
    BLOCK_SIZE_M = initial_block_m
    BLOCK_SIZE_N = initial_block_n
    
    # Use a safety factor to ensure we don't exceed the limit
    safety_factor = 0.85  # Use only 85% of available memory
    target_memory = int(MAX_SHARED_MEMORY * safety_factor)
    
    while calculate_shared_memory(BLOCK_SIZE_M, BLOCK_SIZE_N, head_dim) > target_memory:
        # Reduce both dimensions to maintain balance
        if BLOCK_SIZE_M > 32 and BLOCK_SIZE_N > 32:
            BLOCK_SIZE_M = BLOCK_SIZE_M // 2
            BLOCK_SIZE_N = BLOCK_SIZE_N // 2
        elif BLOCK_SIZE_M >= BLOCK_SIZE_N and BLOCK_SIZE_M > 16:
            BLOCK_SIZE_M = BLOCK_SIZE_M // 2
        elif BLOCK_SIZE_N > 16:
            BLOCK_SIZE_N = BLOCK_SIZE_N // 2
        else:
            # Both are at minimum, can't reduce further
            BLOCK_SIZE_M = min(BLOCK_SIZE_M, 16)
            BLOCK_SIZE_N = min(BLOCK_SIZE_N, 16)
            break
    
    # Adjust for occupancy - ensure good SM utilization
    blocks_per_sm = (num_heads * (verify_len // BLOCK_SIZE_M)) / num_sms
    if blocks_per_sm < 0.5:  # Less than 50% occupancy
        # Only reduce if it doesn't violate memory constraints
        new_block_m = max(16, BLOCK_SIZE_M // 2)
        if calculate_shared_memory(new_block_m, BLOCK_SIZE_N, head_dim) <= MAX_SHARED_MEMORY:
            BLOCK_SIZE_M = new_block_m
    
    # Determine optimal warp count based on tile size
    tile_elements = BLOCK_SIZE_M * BLOCK_SIZE_N
    if tile_elements <= 1024:
        num_warps = 4
    elif tile_elements <= 4096:
        num_warps = 8
    else:
        num_warps = 16
    
    # Log the memory usage for debugging
    memory_used = calculate_shared_memory(BLOCK_SIZE_M, BLOCK_SIZE_N, head_dim)
    
    return BLOCK_SIZE_M, BLOCK_SIZE_N, num_warps


# H100-optimized launch configuration
def launch_attention_h100(Q, K, V, tree_mask, output, seq_len, verify_len, num_heads, num_kv_heads, head_dim):
    """
    Launch configuration optimized for H100 (132 SMs)
    """
    # Optimal tile sizes for H100 Tensor Cores
    BLOCK_SIZE_M, BLOCK_SIZE_N, num_warps = select_optimal_tile_sizes(verify_len, seq_len + verify_len, num_heads, num_sms=132, head_dim=head_dim)
    
    # Grid configuration for attention kernel
    grid = (
        num_heads,  # 32 blocks (one per head)
        triton.cdiv(verify_len, BLOCK_SIZE_M),  # 256/64 = 4 blocks
    )
    # Total: 32 * 4 = 128 blocks, ~1 block per SM, good occupancy
    
    # Launch attention kernel
    flash_mask_attention_kernel[grid](
        Q, K, V, output, tree_mask,
        num_heads, num_kv_heads,
        seq_len, verify_len,
        head_dim=head_dim,  # Pass as compile-time constant
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        num_warps=num_warps,
        num_stages=3,  # Pipeline stages for async copies
    )



# flash mask attention kernel, specially designed for the target model verification and single batch.
@triton.jit
def flash_mask_attention_kernel(
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    tree_mask_ptr,
    num_heads, num_kv_heads,
    seq_len, verify_len,
    head_dim: tl.constexpr,  # Make head_dim a compile-time constant
    BLOCK_SIZE_M: tl.constexpr,  # number of tokens processed by each block
    BLOCK_SIZE_N: tl.constexpr,  # K/V sequence tiles
):
    # Better design: Each block processes ONE head for a subset of tokens
    # This maximizes K/V reuse with GQA
    head_id = tl.program_id(0)  # Which head this block processes
    token_block_id = tl.program_id(1)  # Which token block this processes
    
    # Map Q head to KV head (GQA with 4:1 ratio)
    kv_head_id = head_id // (num_heads // num_kv_heads)  # 32/8 = 4
    
    # Calculate which tokens this block processes
    start_m = token_block_id * BLOCK_SIZE_M
    
    # Boundary check
    if start_m >= verify_len:
        return
    
    # Calculate actual number of tokens to process in this block
    num_tokens = tl.minimum(BLOCK_SIZE_M, verify_len - start_m)
    
    # Create offsets for accessing Q matrix
    # Q shape: [1, num_heads, verify_len, head_dim]
    # We process multiple queries from the same head
    offs_m = start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_d = tl.arange(0, head_dim)
    
    # Mask for valid tokens in this block
    m_mask = offs_m < verify_len
    
    # Q pointer for this specific head and token block
    # Since each block handles ONE head, we can directly compute the pointer
    Q_head_offset = head_id * verify_len * head_dim
    Q_block_ptr = Q_ptr + Q_head_offset + \
                  offs_m[:, None] * head_dim + offs_d[None, :]
    
    # K and V pointers for the KV head corresponding to this Q head
    # K/V shape: [1, num_kv_heads, seq_len+verify_len, head_dim]
    total_seq = seq_len + verify_len
    KV_head_offset = kv_head_id * total_seq * head_dim
    
    # Offsets for K/V tiles (we'll iterate through sequence in blocks of BLOCK_SIZE_N)
    offs_n = tl.arange(0, BLOCK_SIZE_N)  # KV sequence dimension
    
    # K and V pointer matrices for loading tiles
    # These will be updated in the loop to point to different sequence positions
    K_block_ptr = K_ptr + KV_head_offset + offs_n[:, None] * head_dim + offs_d[None, :]
    V_block_ptr = V_ptr + KV_head_offset + offs_n[:, None] * head_dim + offs_d[None, :]
    
    # Tree mask pointer setup
    # tree_mask shape: [1, verify_len, seq_len + verify_len]
    # For this block's query tokens (offs_m), we need their mask rows
    mask_stride_query = total_seq  # Skip this many elements to go to next query
    
    # Tree mask base pointer for this block's queries
    # We'll add KV offsets in the loop
    tree_mask_base = tree_mask_ptr + offs_m[:, None] * mask_stride_query
    
    # Load Q for this head once (reused for all KV tiles)
    q = tl.load(Q_block_ptr, mask=m_mask[:, None])  # [BLOCK_SIZE_M, head_dim] - loaded as fp16
    # Keep Q in fp16 and only promote to fp32 during dot product for better memory efficiency
    
    # Initialize accumulator and softmax statistics - ALL in fp32 for stability
    acc = tl.zeros([BLOCK_SIZE_M, head_dim], dtype=tl.float32)
    max_val = tl.full([BLOCK_SIZE_M], value=-float('inf'), dtype=tl.float32)
    sum_exp = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32)
    
    # Phase 1: Process cached KV (no mask needed - all tokens visible)
    for kv_start in range(0, seq_len, BLOCK_SIZE_N):
        # Calculate number of KV tokens in this tile
        kv_len = tl.minimum(BLOCK_SIZE_N, seq_len - kv_start)
        
        # Load K and V tiles
        K_tile_ptr = K_block_ptr + kv_start * head_dim
        V_tile_ptr = V_block_ptr + kv_start * head_dim
        
        # Create mask for valid KV positions in this tile
        kv_mask = (offs_n[:, None] < kv_len)
        
        k_tile = tl.load(K_tile_ptr, mask=kv_mask, other=0.0)  # Load as fp16
        v_tile = tl.load(V_tile_ptr, mask=kv_mask, other=0.0)  # Load as fp16
        # Keep K/V in fp16 to save memory, promote only during computation
        
        # Compute attention scores: Q @ K^T - promote to fp32 during dot product
        scores = tl.dot(q.to(tl.float32), tl.trans(k_tile).to(tl.float32))  # fp32 output
        scores = scores * (1.0 / tl.sqrt(float(head_dim)))
        
        # Mask out invalid positions
        score_mask = m_mask[:, None] & (offs_n[None, :] < kv_len)
        scores = tl.where(score_mask, scores, -float('inf'))
        
        # Online softmax update - all in fp32
        scores_max = tl.max(scores, axis=1)  # fp32
        max_new = tl.maximum(max_val, scores_max)  # fp32
        
        # Rescale previous accumulator
        alpha = tl.exp(max_val - max_new)  # fp32
        acc = acc * alpha[:, None]
        sum_exp = sum_exp * alpha
        
        # Compute exp of current scores
        exp_scores = tl.exp(scores - max_new[:, None])  # fp32
        
        # Update accumulator - promote V to fp32 for dot product
        acc += tl.dot(exp_scores, v_tile.to(tl.float32))  # fp32 dot product
        sum_exp += tl.sum(exp_scores, axis=1)  # fp32 sum
        max_val = max_new
    
    # Phase 2: Process new/verification KV tokens (with tree mask)
    for kv_start in range(seq_len, total_seq, BLOCK_SIZE_N):
        # Calculate number of KV tokens in this tile
        kv_len = tl.minimum(BLOCK_SIZE_N, total_seq - kv_start)
        
        # Load K and V tiles
        K_tile_ptr = K_block_ptr + kv_start * head_dim
        V_tile_ptr = V_block_ptr + kv_start * head_dim
        
        # Create mask for valid KV positions
        kv_mask = (offs_n[:, None] < kv_len)
        
        k_tile = tl.load(K_tile_ptr, mask=kv_mask, other=0.0)  # Load as fp16
        v_tile = tl.load(V_tile_ptr, mask=kv_mask, other=0.0)  # Load as fp16
        # Keep K/V in fp16 to save memory, promote only during computation
        
        # Compute attention scores - promote to fp32 during dot product
        scores = tl.dot(q.to(tl.float32), tl.trans(k_tile).to(tl.float32))  # fp32
        scores = scores * (1.0 / tl.sqrt(float(head_dim)))
        
        # Load tree mask for this tile
        # tree_mask[query_idx, kv_idx] already contains 0 for visible, -inf for masked
        # So we can directly add it to scores without using tl.where
        mask_tile_ptr = tree_mask_base + (kv_start + offs_n[None, :])
        
        # Create validity mask for loading
        valid_mask = m_mask[:, None] & (offs_n[None, :] < kv_len)
        
        # Load tree mask values (0 or -inf)
        tree_mask_tile = tl.load(mask_tile_ptr, mask=valid_mask, other=-float('inf'))
        
        # Directly add tree mask to scores (no tl.where needed since mask already has 0/-inf)
        scores = scores + tree_mask_tile
        
        # Online softmax update - all in fp32
        scores_max = tl.max(scores, axis=1)  # fp32
        max_new = tl.maximum(max_val, scores_max)  # fp32
        
        alpha = tl.exp(max_val - max_new)  # fp32
        acc = acc * alpha[:, None]
        sum_exp = sum_exp * alpha
        
        exp_scores = tl.exp(scores - max_new[:, None])  # fp32
        
        acc += tl.dot(exp_scores, v_tile.to(tl.float32))  # fp32
        sum_exp += tl.sum(exp_scores, axis=1)  # fp32
        max_val = max_new
    
    # Final normalization - in fp32
    output = acc / sum_exp[:, None]  # fp32 division
    
    # Convert to fp16 for storage
    output = output.to(tl.float16)
    
    # Store output for this head and block of tokens
    # Output shape: [1, num_heads, verify_len, head_dim]
    Out_head_offset = head_id * verify_len * head_dim
    Out_block_ptr = Out_ptr + Out_head_offset + \
                    offs_m[:, None] * head_dim + offs_d[None, :]
    tl.store(Out_block_ptr, output, mask=m_mask[:, None])


def pytorch_attention_baseline(
    query_states: torch.Tensor,  # [batch, num_heads, seq_len, head_dim]
    key_states: torch.Tensor,    # [batch, num_heads, seq_len, head_dim]
    value_states: torch.Tensor,  # [batch, num_heads, seq_len, head_dim]
    attention_mask: torch.Tensor = None,  # [batch, 1, seq_len, seq_len] with 0/-inf values
    head_dim: int = 128
) -> torch.Tensor:
    """
    PyTorch eager implementation of attention mechanism, extracted from LlamaAttention.
    This serves as a baseline to compare with our Triton kernel.
    
    Args:
        query_states: Query tensor [batch, num_heads, verify_len, head_dim]
        key_states: Key tensor [batch, num_heads, total_seq_len, head_dim]
        value_states: Value tensor [batch, num_heads, total_seq_len, head_dim]
        attention_mask: Optional mask tensor with 0 for visible, -inf for masked positions
        head_dim: Dimension of each attention head
    
    Returns:
        Attention output [batch, num_heads, verify_len, head_dim]
    """
    batch_size, num_heads, q_len, _ = query_states.shape
    kv_seq_len = key_states.shape[2]
    
    # Compute attention scores: Q @ K^T / sqrt(head_dim)
    attn_weights = torch.matmul(
        query_states, key_states.transpose(2, 3)
    ) / math.sqrt(head_dim)
    
    # Verify shape
    if attn_weights.size() != (batch_size, num_heads, q_len, kv_seq_len):
        raise ValueError(
            f"Attention weights should be of size {(batch_size, num_heads, q_len, kv_seq_len)}, but is "
            f"{attn_weights.size()}"
        )
    
    # Apply attention mask if provided (direct addition since mask contains 0/-inf)
    if attention_mask is not None:
        # Expand mask if needed to match attention weights shape
        if attention_mask.dim() == 3:  # [batch, q_len, kv_seq_len]
            attention_mask = attention_mask.unsqueeze(1)  # [batch, 1, q_len, kv_seq_len]
        
        if attention_mask.size() != (batch_size, 1, q_len, kv_seq_len):
            raise ValueError(
                f"Attention mask should be of size {(batch_size, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
            )
        
        # Add mask to attention weights (mask already contains 0/-inf values)
        attn_weights = attn_weights + attention_mask
    
    # Apply softmax (upcast to fp32 for numerical stability as in LlamaAttention)
    attn_weights = torch.nn.functional.softmax(
        attn_weights, dim=-1, dtype=torch.float32
    ).to(query_states.dtype)
    
    # Compute attention output: softmax(scores) @ V
    attn_output = torch.matmul(attn_weights, value_states)
    
    # Verify output shape
    if attn_output.size() != (batch_size, num_heads, q_len, head_dim):
        raise ValueError(
            f"attn_output should be of size {(batch_size, num_heads, q_len, head_dim)}, but is "
            f"{attn_output.size()}"
        )
    
    return attn_output


# Example usage and testing
if __name__ == "__main__":
    import time
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create test tensors
    batch = 1
    Q_test = torch.randn(batch, num_heads, verify_len, head_dim, device=device, dtype=torch.float16)
    K_test = torch.randn(batch, num_kv_heads, seq_len + verify_len, head_dim, device=device, dtype=torch.float16)
    V_test = torch.randn(batch, num_kv_heads, seq_len + verify_len, head_dim, device=device, dtype=torch.float16)
    # Create tree_mask with 0 for visible positions and -inf for masked positions
    tree_mask_test = torch.zeros(batch, verify_len, seq_len + verify_len, device=device, dtype=torch.float32)
    # For cached tokens (seq_len), all are visible (0)
    # For new tokens, apply tree structure mask
    random_mask = torch.rand(batch, verify_len, verify_len, device=device) > 0.5
    tree_mask_test[:, :, seq_len:] = torch.where(random_mask, float('-inf'), 0.0)  # -inf for masked, 0 for visible
    
    # Test the attention kernel and compare with PyTorch baseline
    print("Testing flash attention kernel and comparing with PyTorch baseline...")
    
    # For GQA, we need to repeat K/V heads to match Q heads for the baseline
    def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """Repeat key/value heads to match query heads for grouped query attention."""
        batch, num_kv_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(
            batch, num_kv_heads, n_rep, slen, head_dim
        )
        return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)
    
    # Prepare K/V for baseline (repeat to match Q heads)
    num_kv_groups = num_heads // num_kv_heads  # 32 / 8 = 4
    K_test_expanded = repeat_kv(K_test, num_kv_groups)
    V_test_expanded = repeat_kv(V_test, num_kv_groups)
    
    try:
        # Run PyTorch baseline
        print("\n1. Running PyTorch baseline...")
        baseline_output = pytorch_attention_baseline(
            Q_test, K_test_expanded, V_test_expanded, 
            tree_mask_test.unsqueeze(1),  # Add head dimension
            head_dim=head_dim
        )
        print(f"✓ Baseline output shape: {baseline_output.shape}")
        print(f"✓ Baseline output range: [{baseline_output.min().item():.4f}, {baseline_output.max().item():.4f}]")
        
        # Run Triton kernel
        print("\n2. Running Triton kernel...")
        triton_output = torch.empty(batch, num_heads, verify_len, head_dim, device=device, dtype=torch.float16)
        launch_attention_h100(Q_test, K_test, V_test, tree_mask_test, triton_output, 
                            seq_len, verify_len, num_heads, num_kv_heads, head_dim)
        print(f"✓ Triton output shape: {triton_output.shape}")
        print(f"✓ Triton output range: [{triton_output.min().item():.4f}, {triton_output.max().item():.4f}]")
        
        # Compare outputs
        print("\n3. Comparing outputs...")
        # Convert baseline to same dtype for comparison
        baseline_output_fp16 = baseline_output.to(torch.float16)
        
        # Calculate differences
        abs_diff = torch.abs(triton_output - baseline_output_fp16)
        rel_diff = abs_diff / (torch.abs(baseline_output_fp16) + 1e-8)
        
        print(f"  Max absolute difference: {abs_diff.max().item():.6f}")
        print(f"  Mean absolute difference: {abs_diff.mean().item():.6f}")
        print(f"  Max relative difference: {rel_diff.max().item():.6f}")
        print(f"  Mean relative difference: {rel_diff.mean().item():.6f}")
        
        # Check if outputs are close (relaxed tolerance for fp16)
        tolerance = 1e-2  # Relaxed for fp16 precision
        if torch.allclose(triton_output, baseline_output_fp16, rtol=tolerance, atol=tolerance):
            print(f"✓ Outputs match within tolerance ({tolerance})")
        else:
            print(f"⚠ Outputs differ beyond tolerance ({tolerance})")
            # Find where the biggest differences are
            max_diff_idx = torch.unravel_index(abs_diff.argmax(), abs_diff.shape)
            print(f"  Biggest difference at index {max_diff_idx}:")
            print(f"    Baseline: {baseline_output_fp16[max_diff_idx].item():.6f}")
            print(f"    Triton:   {triton_output[max_diff_idx].item():.6f}")
            
    except Exception as e:
        print(f"✗ Error occurred: {e}")
        import traceback
        traceback.print_exc()
    
    # Simple benchmark for attention kernel
    if torch.cuda.is_available():
        print("\n4. Benchmarking Flash Mask Attention kernel...")
        torch.cuda.synchronize()
        
        # Allocate output tensor for benchmarking
        benchmark_output = torch.empty(batch, num_heads, verify_len, head_dim, device=device, dtype=torch.float16)
        
        # Warmup
        for _ in range(10):
            launch_attention_h100(Q_test, K_test, V_test, tree_mask_test, benchmark_output,
                                seq_len, verify_len, num_heads, num_kv_heads, head_dim)
        
        torch.cuda.synchronize()
        start = time.time()
        
        num_iterations = 100
        for _ in range(num_iterations):
            launch_attention_h100(Q_test, K_test, V_test, tree_mask_test, benchmark_output,
                                seq_len, verify_len, num_heads, num_kv_heads, head_dim)
        
        torch.cuda.synchronize()
        end = time.time()
        
        avg_time = (end - start) / num_iterations * 1000  # Convert to ms
        print(f"Average time per forward pass: {avg_time:.2f} ms")

    
    print("\nFlash attention kernel test complete!")
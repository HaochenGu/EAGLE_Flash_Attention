#!/usr/bin/env python3
"""
Test script to verify Flash Attention integration in EAGLE
"""

import torch
from eagle.model.ea_model import EaModel

def test_flash_attention():
    """Test that Flash Attention can be enabled/disabled"""
    
    # Test parameters
    base_model_path = "/path/to/llama31chat/8B/"  # Update this path
    ea_model_path = "/path/to/eagle3/llama31chat/8B/"  # Update this path
    
    print("Testing Flash Attention integration...")
    
    # Test 1: Load model WITHOUT Flash Attention
    print("\n1. Loading model WITHOUT Flash Attention...")
    try:
        model_no_flash = EaModel.from_pretrained(
            base_model_path=base_model_path,
            ea_model_path=ea_model_path,
            total_token=60,
            depth=5,
            top_k=10,
            torch_dtype=torch.float16,
            device_map="auto",
            use_eagle3=True,
            use_flash_attention=False,  # Explicitly disable
        )
        print("✓ Model loaded successfully without Flash Attention")
        
        # Check that standard attention is used
        first_layer = model_no_flash.base_model.model.layers[0]
        attn_class = type(first_layer.self_attn).__name__
        print(f"  Attention class: {attn_class}")
        assert attn_class == "LlamaAttention", f"Expected LlamaAttention, got {attn_class}"
        
    except Exception as e:
        print(f"✗ Failed to load model without Flash Attention: {e}")
        return False
    
    # Test 2: Load model WITH Flash Attention
    print("\n2. Loading model WITH Flash Attention...")
    try:
        model_with_flash = EaModel.from_pretrained(
            base_model_path=base_model_path,
            ea_model_path=ea_model_path,
            total_token=60,
            depth=5,
            top_k=10,
            torch_dtype=torch.float16,
            device_map="auto",
            use_eagle3=True,
            use_flash_attention=True,  # Enable Flash Attention
        )
        print("✓ Model loaded successfully with Flash Attention")
        
        # Check that Triton attention is used
        first_layer = model_with_flash.base_model.model.layers[0]
        attn_class = type(first_layer.self_attn).__name__
        print(f"  Attention class: {attn_class}")
        assert attn_class == "TritonLlamaAttention", f"Expected TritonLlamaAttention, got {attn_class}"
        
    except Exception as e:
        print(f"✗ Failed to load model with Flash Attention: {e}")
        return False
    
    print("\n✅ All tests passed! Flash Attention integration is working correctly.")
    return True

if __name__ == "__main__":
    # Note: Update the model paths before running
    print("Flash Attention Integration Test")
    print("=" * 50)
    print("\nIMPORTANT: Update base_model_path and ea_model_path in the script before running!")
    print("\nTo run the evaluation with Flash Attention, use:")
    print("python gen_ea_answer_llama3chat.py --use_flash_attention [other args...]")
    
    # Uncomment the line below after updating paths:
    # test_flash_attention()
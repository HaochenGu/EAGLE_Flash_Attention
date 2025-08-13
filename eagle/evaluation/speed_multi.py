#!/usr/bin/env python3
"""
Multi-file EAGLE evaluation speed analyzer.
Processes multiple JSONL files and compares their speeds against a baseline.
"""

import json
import numpy as np
import os
import re
from transformers import AutoTokenizer

# ============= CONFIGURATION =============
# Set these parameters according to your setup

# Path to tokenizer
TOKENIZER_PATH = "/home/lyh/weights/hf/llama2chat/13B/"

# Baseline JSONL file
JSONL_FILE_BASE = "llama-2-chat-70b-fp16-base-in-temperature-0.0.jsonl"

# List of EAGLE JSONL files to analyze
# You can use glob patterns or list files explicitly
JSONL_FILES = [
    # Example files - replace with your actual files
    "mt_bench/llama38b_eval-temp-0.0-total_token-64-depth-5-topk-10-flash_attn-True.jsonl",
    "mt_bench/llama38b_eval-temp-0.0-total_token-64-depth-5-topk-10-flash_attn-False.jsonl",
    # Add more files here...
]

# Or use glob to find all matching files
# import glob
# JSONL_FILES = glob.glob("mt_bench/llama38b_eval-*.jsonl")

# ==========================================


def parse_config_from_filename(filename):
    """Extract configuration parameters from filename."""
    config = {}
    
    # Extract total_token
    total_token_match = re.search(r'total_token-([0-9]+)', filename)
    if total_token_match:
        config['total_token'] = total_token_match.group(1)
    
    # Extract depth
    depth_match = re.search(r'depth-([0-9]+)', filename)
    if depth_match:
        config['depth'] = depth_match.group(1)
    
    # Extract top-k
    topk_match = re.search(r'topk-([0-9]+)', filename)
    if topk_match:
        config['top_k'] = topk_match.group(1)
    
    # Extract flash attention
    flash_match = re.search(r'flash_attn-(True|False|true|false)', filename)
    if flash_match:
        config['use_flash_attention'] = flash_match.group(1).capitalize()
    
    return config


def calculate_speed(jsonl_file, tokenizer=None):
    """Calculate average speed for a JSONL file."""
    data = []
    with open(jsonl_file, 'r', encoding='utf-8') as file:
        for line in file:
            json_obj = json.loads(line)
            data.append(json_obj)
    
    speeds = []
    
    for datapoint in data:
        if tokenizer is None:
            # EAGLE output file - has token counts
            tokens = sum(datapoint["choices"][0]['new_tokens'])
        else:
            # Baseline file - need to tokenize
            answer = datapoint["choices"][0]['turns']
            tokens = 0
            for i in answer:
                tokens += (len(tokenizer(i).input_ids) - 1)
        
        times = sum(datapoint["choices"][0]['wall_time'])
        if times > 0:
            speeds.append(tokens / times)
    
    return np.array(speeds).mean() if speeds else 0


def main():
    # Load tokenizer for baseline
    print(f"Loading tokenizer from {TOKENIZER_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    
    # Calculate baseline speed
    print(f"Calculating baseline speed from: {JSONL_FILE_BASE}\n")
    baseline_speed = calculate_speed(JSONL_FILE_BASE, tokenizer)
    
    # Print header
    print(f"{'use_flash_attention':<20} {'topK':<10} {'depth':<10} {'total_token':<15} {'speed ratio':<15}")
    print("-" * 80)
    
    # Process each EAGLE file
    for jsonl_file in JSONL_FILES:
        if not os.path.exists(jsonl_file):
            print(f"Warning: File not found - {jsonl_file}")
            continue
        
        # Extract configuration from filename
        config = parse_config_from_filename(os.path.basename(jsonl_file))
        
        # Calculate speed and ratio
        eagle_speed = calculate_speed(jsonl_file)
        speed_ratio = eagle_speed / baseline_speed if baseline_speed > 0 else 0
        
        # Print results
        flash = config.get('use_flash_attention', 'Unknown')
        topk = config.get('top_k', 'Unknown')
        depth = config.get('depth', 'Unknown')
        total_token = config.get('total_token', 'Unknown')
        
        print(f"{flash:<20} {topk:<10} {depth:<10} {total_token:<15} {speed_ratio:.2f}")


if __name__ == "__main__":
    main()
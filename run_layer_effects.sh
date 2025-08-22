#!/bin/bash

models=(
    # deepseek models
    "DeepSeek-R1-Distill-Qwen-1.5B"
    "DeepSeek-R1-Distill-Qwen-7B"
    "DeepSeek-R1-Distill-Qwen-14B"
    "DeepSeek-R1-Distill-Qwen-32B"
    
    # qwen models
    "Qwen2.5-1.5B-Instruct"
    "Qwen2.5-Math-1.5B"
    "Qwen2.5-7B-Instruct"
    "Qwen2.5-Math-7B"
    "Qwen2.5-14B-Instruct"
    "Qwen2.5-14B"
    "Qwen2.5-32B-Instruct"
    "Qwen2.5-32B"
)

for model in "${models[@]}"; do
    CUDA_VISIBLE_DEVICES=0 python plot_layer_effects.py --model_name "$model" --dataset gsm8k --n_examples 10
done
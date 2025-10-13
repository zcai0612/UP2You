#!/bin/bash

export CUDA_VISIBLE_DEVICES=0


python inference_low_gpu.py \
    --base_model_path stabilityai/stable-diffusion-2-1-base  \
    --shape_predictor_path pretrained_models/shape_predictor.pt \
    --rgb_adapter_path pretrained_models/rgb_adapter.safetensors \
    --feature_aggregator_path pretrained_models/feature_aggregator.pt \
    --normal_adapter_path pretrained_models/normal_adapter.safetensors \
    --segment_model_name ZhengPeng7/BiRefNet  \
    --data_dir examples \
    --output_dir outputs \

# python inference.py \
#     --base_model_path stabilityai/stable-diffusion-2-1-base  \
#     --shape_predictor_path pretrained_models/shape_predictor.pt \
#     --rgb_adapter_path pretrained_models/rgb_adapter.safetensors \
#     --feature_aggregator_path pretrained_models/feature_aggregator.pt \
#     --normal_adapter_path pretrained_models/normal_adapter.safetensors \
#     --segment_model_name ZhengPeng7/BiRefNet  \
#     --data_dir examples \
#     --output_dir outputs \
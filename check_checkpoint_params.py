#!/usr/bin/env python3
"""
Script to analyze checkpoint parameters and help identify the correct model configuration
for resuming training.
"""

import torch
import argparse
import os

def analyze_checkpoint(checkpoint_path):
    """Analyze a checkpoint to understand the model architecture."""
    print(f"Analyzing checkpoint: {checkpoint_path}")
    
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    print("\n=== Checkpoint Analysis ===")
    
    # Check if it's a state dict or has a model key
    if isinstance(checkpoint, dict):
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    # Analyze the state dict
    print(f"Total parameters: {len(state_dict)}")
    
    # Look for key parameters to understand the architecture
    channel_counts = {}
    layer_info = {}
    
    for key, param in state_dict.items():
        if 'weight' in key and len(param.shape) >= 2:
            # This is likely a conv layer
            if 'input_blocks' in key:
                layer_type = 'input'
            elif 'output_blocks' in key:
                layer_type = 'output'
            elif 'middle_block' in key:
                layer_type = 'middle'
            else:
                layer_type = 'other'
            
            # Extract block number if present
            import re
            block_match = re.search(r'\.(\d+)\.', key)
            block_num = block_match.group(1) if block_match else 'unknown'
            
            if layer_type not in layer_info:
                layer_info[layer_type] = {}
            
            if block_num not in layer_info[layer_type]:
                layer_info[layer_type][block_num] = []
            
            layer_info[layer_type][block_num].append({
                'key': key,
                'shape': param.shape,
                'channels': param.shape[0] if len(param.shape) >= 2 else None
            })
    
    # Analyze the architecture
    print("\n=== Architecture Analysis ===")
    
    # Look for input channels
    input_channels = None
    for key, param in state_dict.items():
        if 'input_blocks.0.0.in_layers.0.weight' in key:
            input_channels = param.shape[1]
            print(f"Input channels: {input_channels}")
            break
    
    # Look for model channels (base channel count)
    model_channels = None
    for key, param in state_dict.items():
        if 'input_blocks.0.0.in_layers.0.weight' in key:
            model_channels = param.shape[0]
            print(f"Model channels (base): {model_channels}")
            break
    
    # Analyze channel multipliers
    print("\n=== Channel Analysis ===")
    for layer_type, blocks in layer_info.items():
        print(f"\n{layer_type.upper()} blocks:")
        for block_num, layers in sorted(blocks.items()):
            if layers:
                # Get the first conv layer in this block
                conv_layer = None
                for layer in layers:
                    if 'weight' in layer['key'] and len(layer['shape']) >= 2:
                        conv_layer = layer
                        break
                
                if conv_layer:
                    channels = conv_layer['channels']
                    if model_channels and channels:
                        multiplier = channels / model_channels
                        print(f"  Block {block_num}: {channels} channels (multiplier: {multiplier})")
    
    # Try to infer image size from attention resolutions
    print("\n=== Attention Analysis ===")
    attention_layers = []
    for key in state_dict.keys():
        if 'attn' in key and 'weight' in key:
            attention_layers.append(key)
    
    if attention_layers:
        print(f"Found {len(attention_layers)} attention layers")
        # You can analyze the attention layer shapes to infer resolution
    
    # Check for other important parameters
    print("\n=== Other Parameters ===")
    if 'time_embed' in str(state_dict.keys()):
        print("Time embedding found")
    
    if 'label_emb' in str(state_dict.keys()):
        print("Label embedding found (class conditional)")
    
    # Provide recommendations
    print("\n=== Recommendations ===")
    print("To resume training, ensure these parameters match:")
    if input_channels:
        print(f"- in_channels: {input_channels}")
    if model_channels:
        print(f"- num_channels: {model_channels}")
    
    print("\nCommon issues:")
    print("1. Image size changed (affects channel_mult defaults)")
    print("2. num_channels parameter changed")
    print("3. channel_mult explicitly set differently")
    print("4. learn_sigma parameter changed")
    
    return {
        'input_channels': input_channels,
        'model_channels': model_channels,
        'layer_info': layer_info
    }

def main():
    parser = argparse.ArgumentParser(description="Analyze checkpoint to understand model architecture")
    parser.add_argument("checkpoint_path", help="Path to the checkpoint file")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoint_path):
        print(f"Error: Checkpoint file not found: {args.checkpoint_path}")
        return
    
    analyze_checkpoint(args.checkpoint_path)

if __name__ == "__main__":
    main() 
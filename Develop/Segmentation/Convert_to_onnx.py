# ============================================================================
# IMPORTS AND DEPENDENCIES
# ============================================================================

import argparse
import torch
import torch.onnx
from monai.networks.nets import SegResNet
import os

# ============================================================================
# MODEL SETUP
# ============================================================================

def create_model(in_channels=4, num_classes=3, init_filters=16):
    """Recreate the SegResNet model matching training config"""
    model = SegResNet(
        blocks_down=[1, 2, 2, 4],
        blocks_up=[1, 1, 1],
        init_filters=init_filters,
        in_channels=in_channels,
        out_channels=num_classes,
        dropout_prob=0.2,
    )
    return model

# ============================================================================
# CONVERSION UTILITIES
# ============================================================================

def convert_to_onnx(checkpoint_path, output_path, roi_size=(128, 128, 128)):
    """Load .pth checkpoint and export to ONNX with correct dummy shape (160x160x128) and dynamic axes"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    device = torch.device('cpu') 
    
    # Recreate model
    model = create_model()
    model.load_state_dict(torch.load(checkpoint_path, map_location=device)['model_state_dict'])
    model.eval()
    
    dummy_input = torch.randn(1, 4, 160, 160, 128)  # Dummy input for ONNX export
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size', 2: 'height', 3: 'width', 4: 'depth'},
            'output': {0: 'batch_size', 2: 'height', 3: 'width', 4: 'depth'}
        }
    )
    
    print(f"✓ ONNX model exported: {output_path}")
    print(f"  Model size: {os.path.getsize(output_path) / (1024**2):.2f} MB")
    print(f"  Input shape: [batch, 4, height(~160), width(~160), depth(128)] (dynamic spatial)")

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert SegResNet .pth to ONNX")
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to .pth checkpoint')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save .onnx')
    parser.add_argument('--roi_size', type=int, nargs=3, default=[128, 128, 128], help='ROI size (used for depth pad)')
    args = parser.parse_args()
    args.roi_size = tuple(args.roi_size)
    convert_to_onnx(args.checkpoint_path, args.output_path, args.roi_size)
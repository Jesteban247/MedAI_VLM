"""Image processing utilities"""

import os
import base64
import tempfile
from io import BytesIO
from PIL import Image
from typing import List, Tuple, Optional


def get_images_from_folder(folder_path: str) -> List[str]:
    """Get all image files from a folder"""
    if not os.path.exists(folder_path):
        return []
    
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
    return [
        os.path.join(folder_path, f) 
        for f in sorted(os.listdir(folder_path))
        if f.lower().endswith(valid_extensions)
    ]


def get_seg3d_folders() -> List[str]:
    """Get all case folders from Seg_3D directory"""
    seg3d_path = "Files_Seg3D"
    if not os.path.exists(seg3d_path):
        return []
    
    return [
        f for f in sorted(os.listdir(seg3d_path)) 
        if os.path.isdir(os.path.join(seg3d_path, f))
    ]


def base64_to_image(b64_string: str) -> Image.Image:
    """Convert base64 string to PIL Image"""
    img_data = base64.b64decode(b64_string)
    return Image.open(BytesIO(img_data))


def save_image_to_temp(image: Image.Image) -> str:
    """Save PIL Image to temporary file and return path"""
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        image.save(f.name, format='PNG')
        return f.name


def process_cam_visualizations(cam_viz: dict) -> Tuple[List[Image.Image], List[str]]:
    """
    Process CAM visualizations from result
    
    Returns:
        Tuple of (images, labels)
    """
    if not cam_viz:
        return [], []
    
    # Sort by method priority
    method_priority = {'GradCAM++': 0, 'EigenCAM': 1, 'LayerCAM': 2, 'GradCAM': 3}
    sorted_viz = sorted(
        cam_viz.items(), 
        key=lambda x: (method_priority.get(x[0].split('_')[0], 99), x[0])
    )
    
    viz_images = []
    viz_labels = []
    
    for name, img_b64 in sorted_viz:
        try:
            img = base64_to_image(img_b64)
            viz_images.append(img)
            
            # Format label: "GradCAM++ (conv_1x1_exp)"
            method, layer = name.split('_', 1)
            viz_labels.append(f"{method} ({layer.replace('_', ' ')})")
        except Exception:
            continue  # Skip failed visualizations
    
    return viz_images, viz_labels


def save_images_to_temp(images: List[Image.Image]) -> List[str]:
    """Save multiple PIL Images to temporary files"""
    paths = []
    for img in images:
        if isinstance(img, str):
            paths.append(img)
        else:
            paths.append(save_image_to_temp(img))
    return paths

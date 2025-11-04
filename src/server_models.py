"""Medical AI models server (Classification, Detection, Segmentation)"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import sys
import json
from datetime import datetime
import uvicorn
import warnings
import base64
from io import BytesIO

# Handle both direct execution and module import
if __name__ == "__main__":
    # Direct execution: add parent directory to path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.logger import setup_logger
    from src.config import (
        DOWNSAMPLE_FACTOR, MAX_BRAIN_POINTS, MAX_TUMOR_POINTS,
        SLICE_OFFSETS_3, SLICE_OFFSETS_5
    )
else:
    # Module import: use relative imports
    from .logger import setup_logger
    from .config import (
        DOWNSAMPLE_FACTOR, MAX_BRAIN_POINTS, MAX_TUMOR_POINTS,
        SLICE_OFFSETS_3, SLICE_OFFSETS_5
    )

warnings.filterwarnings("ignore")
logger = setup_logger(__name__)

app = FastAPI(title="Medical AI Models Server")

DETECTION_MODELS = {
    'Blood_Cell': 'Models/Detection/Blood_Cell.onnx',
    'Breast_Cancer': 'Models/Detection/Breast_Cancer.onnx', 
    'Fracture': 'Models/Detection/Fracture.onnx'
}

CLASSIFICATION_MODELS = {
    'Brain_Tumor': 'Models/Classification/brain_tumor',
    'Chest_X-Ray': 'Models/Classification/chest-xray',
    'Lung_Cancer': 'Models/Classification/lung-cancer'
}

SEGMENTATION_MODELS = {
    'brats': 'Models/Seg_3D/Brats.onnx'
}

class DetectionRequest(BaseModel):
    image_path: str
    model: str

class ClassificationRequest(BaseModel):
    image_path: str
    model: str

class SegmentationRequest(BaseModel):
    case_path: str
    model: str = "brats"

detection_models = {}  # Cache for loaded YOLO detection models
classification_models_cache = {}  # Cache for loaded classification models


def load_detection_models():
    """Pre-load YOLO detection models"""
    from ultralytics import YOLO
    for model_name, model_path in DETECTION_MODELS.items():
        if os.path.exists(model_path):
            detection_models[model_name] = YOLO(model_path, task='detect')
            logger.info(f"Loaded detection model: {model_name}")
        else:
            logger.warning(f"Detection model not found: {model_path}")

def load_all_classification_models():
    logger.info("Pre-loading classification models...")
    for model_name, model_path in CLASSIFICATION_MODELS.items():
        if os.path.exists(model_path):
            try:
                get_classification_model(model_name)
                logger.info(f"Pre-loaded classification model: {model_name}")
            except Exception as e:
                logger.warning(f"Failed to pre-load {model_name}: {e}")
        else:
            logger.warning(f"Classification model not found: {model_path}")

def image_to_base64(image):
    """Convert PIL Image to base64"""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


def figure_to_base64(fig):
    """Convert matplotlib figure to base64"""
    import matplotlib.pyplot as plt
    buffered = BytesIO()
    fig.savefig(buffered, format='PNG', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buffered.seek(0)
    return base64.b64encode(buffered.read()).decode()

def load_classification_imports():
    """Lazy load classification dependencies"""
    try:
        import torch
        import numpy as np
        import matplotlib.pyplot as plt
        from PIL import Image
        from peft import PeftModel, PeftConfig
        from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, EigenCAM, LayerCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        from transformers import AutoImageProcessor, AutoModelForImageClassification
        from torchvision.transforms import Compose, Normalize, Resize, CenterCrop, ToTensor
        return {
            'torch': torch, 'np': np, 'plt': plt, 'Image': Image,
            'PeftModel': PeftModel, 'PeftConfig': PeftConfig,
            'GradCAM': GradCAM, 'GradCAMPlusPlus': GradCAMPlusPlus, 
            'EigenCAM': EigenCAM, 'LayerCAM': LayerCAM,
            'show_cam_on_image': show_cam_on_image,
            'ClassifierOutputTarget': ClassifierOutputTarget,
            'AutoImageProcessor': AutoImageProcessor,
            'AutoModelForImageClassification': AutoModelForImageClassification,
            'Compose': Compose, 'Normalize': Normalize, 'Resize': Resize,
            'CenterCrop': CenterCrop, 'ToTensor': ToTensor
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification dependencies unavailable: {e}")

def load_segmentation_imports():
    """Lazy load segmentation dependencies"""
    try:
        import torch
        import numpy as np
        import nibabel as nib
        import onnxruntime as ort
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
        from matplotlib.colors import ListedColormap
        from monai.transforms import (
            Compose, LoadImaged, EnsureChannelFirstd, EnsureTyped, 
            Orientationd, Spacingd, NormalizeIntensityd, SpatialPadd
        )
        from monai.data import Dataset, DataLoader
        return {
            'torch': torch, 'np': np, 'nib': nib, 'ort': ort, 'plt': plt,
            'Patch': Patch, 'ListedColormap': ListedColormap,
            'Compose': Compose, 'LoadImaged': LoadImaged,
            'EnsureChannelFirstd': EnsureChannelFirstd, 'EnsureTyped': EnsureTyped,
            'Orientationd': Orientationd, 'Spacingd': Spacingd,
            'NormalizeIntensityd': NormalizeIntensityd, 'SpatialPadd': SpatialPadd,
            'Dataset': Dataset, 'DataLoader': DataLoader
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Segmentation dependencies unavailable: {e}")

def get_classification_model(model_name):
    """Load or retrieve cached classification model"""
    # Return cached model if available
    if model_name in classification_models_cache:
        return classification_models_cache[model_name]
    
    # Validate model exists
    if model_name not in CLASSIFICATION_MODELS:
        raise HTTPException(status_code=400, detail=f"Model {model_name} not available")
    
    model_path = CLASSIFICATION_MODELS[model_name]
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"Model path not found: {model_path}")
    
    libs = load_classification_imports()
    
    try:
        # Check if model is a LoRA adapter or full model
        is_lora_adapter = os.path.exists(os.path.join(model_path, "adapter_config.json"))
        
        if is_lora_adapter:
            model_info = _load_lora_model(model_path, libs)
        else:
            model_info = _load_full_model(model_path, libs)
        
        # Cache model components for reuse
        classification_models_cache[model_name] = model_info
        logger.info(f"Loaded classification model: {model_name}")
        return model_info
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model {model_name}: {e}")


def _load_lora_model(model_path, libs):
    """Load LoRA adapter model with base model"""
    peft_cfg = libs['PeftConfig'].from_pretrained(model_path)
    parent_dir = os.path.dirname(model_path)
    cfg_path = os.path.join(parent_dir, "config.json")
    
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Missing config.json in {parent_dir}")
    
    with open(cfg_path, "r") as f:
        cfg = json.load(f)
    
    class_names = cfg.get("classes", [])
    num_classes = len(class_names)
    
    model = libs['AutoModelForImageClassification'].from_pretrained(
        peft_cfg.base_model_name_or_path,
        num_labels=num_classes,
        label2id={cls: i for i, cls in enumerate(class_names)},
        id2label={i: cls for i, cls in enumerate(class_names)},
        ignore_mismatched_sizes=True,
        device_map="auto"
    )
    model = libs['PeftModel'].from_pretrained(model, model_path)
    processor = libs['AutoImageProcessor'].from_pretrained(peft_cfg.base_model_name_or_path)
    
    return {'model': model, 'processor': processor, 'class_names': class_names}


def _load_full_model(model_path, libs):
    """Load full fine-tuned model"""
    model = libs['AutoModelForImageClassification'].from_pretrained(model_path, device_map="auto")
    processor = libs['AutoImageProcessor'].from_pretrained(model_path)
    class_names = [model.config.id2label[i] for i in range(len(model.config.id2label))]
    
    return {'model': model, 'processor': processor, 'class_names': class_names}

@app.post("/detect")
async def detect_objects(request: DetectionRequest):
    """Run YOLO detection on medical image"""
    if request.model not in detection_models:
        raise HTTPException(status_code=400, detail=f"Model {request.model} not available")
        
    if not os.path.exists(request.image_path):
        raise HTTPException(status_code=404, detail=f"Image not found: {request.image_path}")
    
    from PIL import Image
    model = detection_models[request.model]
    
    # Run YOLO inference
    results = model.predict(source=request.image_path, imgsz=640, verbose=False)
    
    # Extract predictions from results
    predictions = []
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                pred = {
                    'bbox': box.xyxy[0].tolist(),  # [x1, y1, x2, y2]
                    'confidence': float(box.conf[0]),
                    'class': int(box.cls[0]),
                    'class_name': result.names[int(box.cls[0])] if result.names else str(int(box.cls[0]))
                }
                predictions.append(pred)
    
    # Generate annotated image with bounding boxes
    annotated_img = results[0].plot()
    # YOLO plot() returns BGR format, convert to RGB for PIL
    annotated_img_rgb = annotated_img[:, :, ::-1]
    annotated_img_pil = Image.fromarray(annotated_img_rgb)
    annotated_base64 = image_to_base64(annotated_img_pil)
    
    return {
        'task': 'detection',
        'model_used': request.model,
        'image_path': request.image_path,
        'timestamp': datetime.now().isoformat(),
        'predictions': predictions,
        'total_detections': len(predictions),
        'annotated_image': annotated_base64
    }

def get_label(model, idx):
    """Get class label by index"""
    id2label = model.config.id2label
    key = str(idx) if str(idx) in id2label else idx
    return id2label[key]

def preprocess_image_for_classification(image, processor, libs):
    """Preprocess image using training pipeline (Train.py)"""
    # Determine image size
    size = _get_processor_size(processor)
    
    # Get normalization parameters
    normalize = _get_normalization(processor, libs)
    
    # Apply transforms
    transforms = libs['Compose']([
        libs['Resize'](size),
        libs['CenterCrop'](size),
        libs['ToTensor'](),
        normalize
    ])
    return transforms(image)


def _get_processor_size(processor):
    """Extract image size from processor"""
    if not hasattr(processor, 'size'):
        return 224
    
    if isinstance(processor.size, dict):
        return processor.size.get("shortest_edge", 224)
    
    return processor.size


def _get_normalization(processor, libs):
    """Get normalization transform from processor or use defaults"""
    if hasattr(processor, 'image_mean') and hasattr(processor, 'image_std'):
        return libs['Normalize'](mean=processor.image_mean, std=processor.image_std)
    
    # Default ImageNet normalization
    return libs['Normalize'](mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def generate_multiple_cam_visualizations(model, image, processor, class_names, probs, top_idx, top_label, top_prob, libs):
    """
    Generate multiple CAM visualizations (GradCAM, GradCAM++, EigenCAM, LayerCAM)
    
    Uses 3 optimal target layers for MobileViT:
    1. conv_1x1_exp: Final conv layer (highest-level features, most reliable)
    2. fusion: CNN+Transformer fusion (captures both modalities)
    3. conv_projection: After transformer (attention visualization)
    
    Excludes ScoreCAM (too slow for production)
    """
    # Use processor for CAM
    inputs = processor(images=image, return_tensors="pt")
    img_tensor = inputs["pixel_values"].squeeze(0)
    
    # Ensure tensor is on same device as model
    device = next(model.parameters()).device
    img_tensor = img_tensor.to(device)
    
    # Wrapper for CAM methods
    class HuggingfaceToTensorModelWrapper(libs['torch'].nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, x):
            return self.model(x).logits
    
    model.eval()
    targets = [libs['ClassifierOutputTarget'](top_idx)]
    wrapper = HuggingfaceToTensorModelWrapper(model)
    
    # Determine if PEFT model
    is_peft = isinstance(model, libs['PeftModel'])
    base_model = model.base_model.model if is_peft else model
    
    # Define 3 optimal target layers for MobileViT (ordered by importance)
    target_layers = {}
    
    # Layer 1: Final conv layer (MOST IMPORTANT - always works well)
    target_layers['conv_1x1_exp'] = base_model.mobilevit.conv_1x1_exp
    
    # Layer 2: Last fusion layer (CNN+Transformer combination)
    if hasattr(base_model.mobilevit.encoder.layer[-1], 'fusion'):
        target_layers['last_fusion'] = base_model.mobilevit.encoder.layer[-1].fusion
    
    # Layer 3: Last conv projection (after transformer processing)
    if hasattr(base_model.mobilevit.encoder.layer[-1], 'conv_projection'):
        target_layers['last_conv_proj'] = base_model.mobilevit.encoder.layer[-1].conv_projection
    
    # CAM methods (4 fast methods, excluding ScoreCAM)
    cam_methods = {
        'GradCAM': libs['GradCAM'],
        'GradCAM++': libs['GradCAMPlusPlus'],
        'EigenCAM': libs['EigenCAM'],
        'LayerCAM': libs['LayerCAM']
    }
    
    # Generate CAM visualizations (4 methods × 3 layers = 12 total)
    cam_results = {}
    rgb_img = libs['np'].float32(image.resize((256, 256))) / 255.0
    
    for layer_name, target_layer in target_layers.items():
        for method_name, cam_class in cam_methods.items():
            try:
                with cam_class(model=wrapper, target_layers=[target_layer]) as cam:
                    grayscale_cam = cam(input_tensor=img_tensor.unsqueeze(0), targets=targets)[0, :]
                
                # Create visualization
                cam_img = libs['show_cam_on_image'](rgb_img, grayscale_cam, use_rgb=True)
                
                # Create figure for this combination
                fig, axes = libs['plt'].subplots(1, 2, figsize=(12, 5))
                axes[0].imshow(image)
                axes[0].set_title("Original Image", fontsize=12, fontweight='bold')
                axes[0].axis("off")
                axes[1].imshow(cam_img)
                axes[1].set_title(f"{method_name} ({layer_name}): {top_label} ({top_prob*100:.2f}%)", 
                                fontsize=11, fontweight='bold')
                axes[1].axis("off")
                
                legend_text = " | ".join([f"{cls}: {p*100:.2f}%" for cls, p in zip(class_names, probs)])
                libs['plt'].figtext(0.5, 0.02, legend_text, ha="center", fontsize=10, style='italic')
                libs['plt'].tight_layout(rect=[0, 0.08, 1, 1])
                
                # Store result
                key = f"{method_name}_{layer_name}"
                cam_results[key] = figure_to_base64(fig)
                libs['plt'].close(fig)
                
            except Exception as e:
                logger.warning(f"Failed to generate {method_name} with {layer_name}: {e}")
                continue
    
    return cam_results

@app.post("/classify")
async def classify_image(request: ClassificationRequest):
    """Classification with multiple CAM visualizations (GradCAM, GradCAM++, EigenCAM, LayerCAM)"""
    if not os.path.exists(request.image_path):
        raise HTTPException(status_code=404, detail=f"Image not found: {request.image_path}")
    
    libs = load_classification_imports()
    model_info = get_classification_model(request.model)
    model = model_info['model']
    processor = model_info['processor']
    class_names = model_info['class_names']
    
    image = libs['Image'].open(request.image_path).convert("RGB")
    
    # Use Train.py preprocessing for accurate predictions
    img_tensor = preprocess_image_for_classification(image, processor, libs)
    
    # Ensure tensor is on same device as model
    device = next(model.parameters()).device
    img_tensor = img_tensor.to(device)
    
    with libs['torch'].no_grad():
        logits = model(img_tensor.unsqueeze(0)).logits
        probs = libs['torch'].softmax(logits, dim=-1)[0].cpu().numpy()
    
    top_idx = int(libs['np'].argmax(probs))
    top_label = class_names[top_idx]
    top_prob = probs[top_idx]
    
    predictions = [
        {'class_name': class_name, 'confidence': float(prob), 'class_id': i}
        for i, (class_name, prob) in enumerate(zip(class_names, probs))
    ]
    predictions.sort(key=lambda x: x['confidence'], reverse=True)
    
    # Generate multiple CAM visualizations
    cam_visualizations = {}
    try:
        cam_visualizations = generate_multiple_cam_visualizations(
            model, image, processor, class_names, probs, top_idx, top_label, top_prob, libs
        )
        logger.info(f"Generated {len(cam_visualizations)} CAM visualizations")
    except Exception as e:
        logger.error(f"CAM generation failed: {str(e)}")
        logger.exception("Full error:")
    
    return {
        'task': 'classification',
        'model_used': request.model,
        'image_path': request.image_path,
        'timestamp': datetime.now().isoformat(),
        'top_prediction': {'class_name': top_label, 'confidence': float(top_prob)},
        'all_predictions': predictions,
        'class_names': class_names,
        'cam_visualizations': cam_visualizations,
        'note': f'Generated {len(cam_visualizations)} CAM visualizations' if cam_visualizations else 'CAM generation failed'
    }

class ConvertToMultiChannelBasedOnBratsClassesd:
    """Convert BraTS labels to multi-channel (TC, WT, ET)"""
    def __init__(self, keys):
        self.keys = [keys] if isinstance(keys, str) else keys
        
    def __call__(self, data):
        libs = load_segmentation_imports()
        d = dict(data)
        for key in self.keys:
            result = []
            # Channel 0: Tumor Core (TC) = NCR/NET (1) or ET (4)
            result.append(libs['torch'].logical_or(d[key] == 1, d[key] == 4))
            # Channel 1: Whole Tumor (WT) = TC + ED (2)
            result.append(libs['torch'].logical_or(libs['torch'].logical_or(d[key] == 1, d[key] == 4), d[key] == 2))
            # Channel 2: Enhancing Tumor (ET) = ET (4) only
            result.append(d[key] == 4)
            d[key] = libs['torch'].stack(result, axis=0).float()
        return d

def get_segmentation_transforms(roi_size=(128, 128, 128)):
    """Create MONAI preprocessing pipeline"""
    libs = load_segmentation_imports()
    return libs['Compose']([
        libs['LoadImaged'](keys=["image", "label"]),  # Load NIfTI files
        libs['EnsureChannelFirstd'](keys="image"),  # Ensure channel-first format
        libs['EnsureTyped'](keys=["image", "label"]),  # Convert to tensors
        ConvertToMultiChannelBasedOnBratsClassesd(keys=["label"]),  # Convert labels to multi-channel
        libs['Orientationd'](keys=["image", "label"], axcodes="RAS"),  # Standardize orientation
        libs['Spacingd'](keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),  # Resample
        libs['SpatialPadd'](keys=["image", "label"], spatial_size=roi_size, mode="constant"),  # Pad to fixed size
        libs['NormalizeIntensityd'](keys="image", nonzero=True, channel_wise=True),  # Normalize intensities
    ])

def calculate_dice_score(pred, target, epsilon=1e-8):
    """Calculate Dice coefficient per channel"""
    dice_scores = []
    for c in range(pred.shape[0]):
        pred_c = pred[c].flatten()
        target_c = target[c].flatten()
        intersection = (pred_c * target_c).sum()
        dice = (2.0 * intersection + epsilon) / (pred_c.sum() + target_c.sum() + epsilon)
        dice_scores.append(dice.item())
    return dice_scores

def visualize_segmentation(pred_data, libs):
    """Create segmentation visualization with MRI modalities"""
    case_id = pred_data['case_id']
    image = pred_data['image']
    pred = pred_data['pred']
    label = pred_data['label']
    dice = pred_data['dice']
    
    # Find slice with most tumor for visualization
    tumor_per_slice = label.sum(axis=(0, 1, 2))
    slice_idx = int(libs['np'].argmax(tumor_per_slice))
    if tumor_per_slice[slice_idx] == 0:
        slice_idx = label.shape[3] // 2  # Use middle slice if no tumor
    
    # Extract all MRI modalities for selected slice
    flair = image[0, :, :, slice_idx]
    t1 = image[1, :, :, slice_idx]
    t1ce = image[2, :, :, slice_idx]
    t2 = image[3, :, :, slice_idx]
    
    def multi_to_single(multi_slice):
        """Convert multi-channel segmentation to single-channel with BraTS labels"""
        single = libs['np'].zeros_like(multi_slice[0])
        et_mask = multi_slice[2] > 0.5  # ET = 4
        tc_mask = (multi_slice[0] > 0.5) & (~et_mask)  # NCR/NET = 1
        wt_mask = (multi_slice[1] > 0.5) & (libs['np'].logical_not(multi_slice[0] > 0.5))  # ED = 2
        single[et_mask] = 4
        single[tc_mask] = 1
        single[wt_mask] = 2
        return single
    
    # Convert multi-channel to single-channel for visualization
    label_slice = multi_to_single(label[:, :, :, slice_idx])
    pred_slice = multi_to_single(pred[:, :, :, slice_idx])
    
    # Create 2x4 grid: top row = modalities, bottom row = segmentations
    fig, axes = libs['plt'].subplots(2, 4, figsize=(24, 12))
    
    # Add title with case info and Dice scores
    title = f"{case_id} - Axial Slice {slice_idx} (Tumor Vol: {label.sum(axis=(0,1,2))[slice_idx]:.0f} vox)\n"
    title += f"Dice: TC={dice[0]:.3f}, WT={dice[1]:.3f}, ET={dice[2]:.3f} | Avg={libs['np'].mean(dice):.3f}"
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    axes[0, 0].imshow(flair.T, cmap='gray', origin='lower')
    axes[0, 0].set_title('FLAIR', fontsize=12); axes[0, 0].axis('off')
    axes[0, 1].imshow(t1.T, cmap='gray', origin='lower')
    axes[0, 1].set_title('T1', fontsize=12); axes[0, 1].axis('off')
    axes[0, 2].imshow(t1ce.T, cmap='gray', origin='lower')
    axes[0, 2].set_title('T1CE', fontsize=12); axes[0, 2].axis('off')
    axes[0, 3].imshow(t2.T, cmap='gray', origin='lower')
    axes[0, 3].set_title('T2', fontsize=12); axes[0, 3].axis('off')
    
    # Bottom row: Segmentation results with color-coded labels
    colors = ['black', 'red', 'green', 'white', 'blue']  # 0=bg, 1=NCR, 2=ED, 3=unused, 4=ET
    cmap = libs['ListedColormap'](colors)
    axes[1, 0].imshow(label_slice.T, cmap=cmap, origin='lower', vmin=0, vmax=4)
    axes[1, 0].set_title('Ground Truth', fontsize=12, fontweight='bold'); axes[1, 0].axis('off')
    axes[1, 1].imshow(pred_slice.T, cmap=cmap, origin='lower', vmin=0, vmax=4)
    axes[1, 1].set_title('Prediction', fontsize=12, fontweight='bold'); axes[1, 1].axis('off')
    
    # Overlay: Show matches (green) and errors (red)
    axes[1, 2].imshow(t1ce.T, cmap='gray', origin='lower', alpha=0.7)
    match_mask = (label_slice == pred_slice) & (label_slice > 0)
    error_mask = (label_slice != pred_slice) & ((label_slice > 0) | (pred_slice > 0))
    axes[1, 2].imshow(match_mask.T.astype(float), cmap='Greens', alpha=0.6, origin='lower')
    axes[1, 2].imshow(error_mask.T.astype(float), cmap='Reds', alpha=0.6, origin='lower')
    axes[1, 2].set_title('Overlay (Green=Match, Red=Error)', fontsize=12, fontweight='bold'); axes[1, 2].axis('off')
    axes[1, 3].axis('off')
    
    legend_elements = [libs['Patch'](facecolor='black', label='0: Background'),
                      libs['Patch'](facecolor='red', label='1: NCR/NET (TC)'),
                      libs['Patch'](facecolor='green', label='2: ED (WT)'),
                      libs['Patch'](facecolor='blue', label='4: ET')]
    fig.legend(handles=legend_elements, loc='lower center', fontsize='medium', ncol=4, bbox_to_anchor=(0.5, -0.02))
    libs['plt'].tight_layout(rect=[0, 0.08, 1, 1])
    
    return figure_to_base64(fig)

def generate_additional_visualizations(pred_data, libs):
    """Generate multi-slice visualizations (1, 3, 5 slices)"""
    case_id = pred_data['case_id']
    image = pred_data['image']
    pred = pred_data['pred']
    total_slices = image.shape[3]
    
    # Find slice with maximum tumor segmentation
    tumor_per_slice = pred.sum(axis=(0, 1, 2))
    max_slice = int(libs['np'].argmax(tumor_per_slice))
    if tumor_per_slice[max_slice] == 0:
        max_slice = total_slices // 2
    
    def multi_to_single(multi_slice):
        """Convert multi-channel to single-channel with BraTS labels"""
        single = libs['np'].zeros_like(multi_slice[0])
        et_mask = multi_slice[2] > 0.5
        tc_mask = (multi_slice[0] > 0.5) & (~et_mask)
        wt_mask = (multi_slice[1] > 0.5) & (libs['np'].logical_not(multi_slice[0] > 0.5))
        single[et_mask] = 4
        single[tc_mask] = 1
        single[wt_mask] = 2
        return single
    
    # Color legend for all visualizations
    legend_elements = [libs['plt'].Rectangle((0,0),1,1, facecolor='red', alpha=0.5, label='NCR/NET'),
                      libs['plt'].Rectangle((0,0),1,1, facecolor='green', alpha=0.5, label='ED'),
                      libs['plt'].Rectangle((0,0),1,1, facecolor='blue', alpha=0.5, label='ET')]
    
    visualization_b64 = {}
    
    # 1. Single slice visualization
    fig, ax = libs['plt'].subplots(1, 1, figsize=(6, 6))
    fig.suptitle(f'{case_id} - Single Most Segmented Slice', fontsize=12, fontweight='bold')
    img = image[2, :, :, max_slice]  # Use T1CE modality
    pred_slice = multi_to_single(pred[:, :, :, max_slice])
    ax.imshow(libs['np'].rot90(img, k=2), cmap='gray')
    # Overlay each tumor class with different color
    for lbl, color in [(1, 'red'), (2, 'green'), (4, 'blue')]:
        mask = pred_slice == lbl
        ax.imshow(libs['np'].rot90(mask, k=2), cmap=libs['ListedColormap'](['none', color]), alpha=0.5)
    ax.set_title(f'Slice {max_slice}')
    ax.axis('off')
    fig.legend(handles=legend_elements, loc='lower center', fontsize='small', ncol=3, bbox_to_anchor=(0.5, -0.05))
    libs['plt'].tight_layout()
    visualization_b64['single_slice'] = figure_to_base64(fig)
    
    # 2. Three-slice visualization (offset by 10)
    offsets = SLICE_OFFSETS_3
    slices_3 = []
    for offset in offsets:
        idx_val = max_slice + offset
        if 0 <= idx_val < total_slices:
            slices_3.append(idx_val)
    # Fill to 3 slices if needed
    while len(slices_3) < 3:
        if slices_3[0] > 0:
            slices_3.insert(0, slices_3[0] - 1)
        elif slices_3[-1] < total_slices - 1:
            slices_3.append(slices_3[-1] + 1)
        else:
            break
    slices_3 = slices_3[:3]
    
    fig, axes = libs['plt'].subplots(1, 3, figsize=(24, 8))
    fig.suptitle(f'{case_id} - 3 Slices (Jump=10)', fontsize=18, fontweight='bold', y=0.98)
    for i, slice_idx in enumerate(slices_3):
        img = image[2, :, :, slice_idx]
        pred_slice = multi_to_single(pred[:, :, :, slice_idx])
        axes[i].imshow(libs['np'].rot90(img, k=2), cmap='gray')
        for lbl, color in [(1, 'red'), (2, 'green'), (4, 'blue')]:
            mask = pred_slice == lbl
            axes[i].imshow(libs['np'].rot90(mask, k=2), cmap=libs['ListedColormap'](['none', color]), alpha=0.5)
        axes[i].set_title(f'Slice {slice_idx}', fontsize=16, fontweight='bold', pad=15)
        axes[i].axis('off')
    fig.legend(handles=legend_elements, loc='lower center', fontsize='large', ncol=3, bbox_to_anchor=(0.5, -0.01))
    libs['plt'].tight_layout(rect=[0, 0.03, 1, 0.96])
    visualization_b64['three_slices'] = figure_to_base64(fig)
    
    # 3. Five-slice visualization (offset by 5)
    offsets = SLICE_OFFSETS_5
    slices_5 = []
    for offset in offsets:
        idx_val = max_slice + offset
        if 0 <= idx_val < total_slices:
            slices_5.append(idx_val)
    # Fill to 5 slices if needed
    while len(slices_5) < 5:
        if slices_5[0] > 0:
            slices_5.insert(0, slices_5[0] - 1)
        elif slices_5[-1] < total_slices - 1:
            slices_5.append(slices_5[-1] + 1)
        else:
            break
    slices_5 = slices_5[:5]
    
    fig, axes = libs['plt'].subplots(1, 5, figsize=(30, 6))
    fig.suptitle(f'{case_id} - 5 Slices (Jump=5)', fontsize=18, fontweight='bold', y=0.98)
    for i, slice_idx in enumerate(slices_5):
        img = image[2, :, :, slice_idx]
        pred_slice = multi_to_single(pred[:, :, :, slice_idx])
        axes[i].imshow(libs['np'].rot90(img, k=2), cmap='gray')
        for lbl, color in [(1, 'red'), (2, 'green'), (4, 'blue')]:
            mask = pred_slice == lbl
            axes[i].imshow(libs['np'].rot90(mask, k=2), cmap=libs['ListedColormap'](['none', color]), alpha=0.5)
        axes[i].set_title(f'Slice {slice_idx}', fontsize=14, fontweight='bold', pad=12)
        axes[i].axis('off')
    fig.legend(handles=legend_elements, loc='lower center', fontsize='large', ncol=3, bbox_to_anchor=(0.5, -0.01))
    libs['plt'].tight_layout(rect=[0, 0.03, 1, 0.96])
    visualization_b64['five_slices'] = figure_to_base64(fig)
    
    return visualization_b64

def create_3d_prediction_html(pred_data, libs, downsample_factor=None, max_brain_points=None, max_tumor_points=None):
    """Create interactive 3D Plotly visualization"""
    # Use config values if not provided
    if downsample_factor is None:
        downsample_factor = DOWNSAMPLE_FACTOR
    if max_brain_points is None:
        max_brain_points = MAX_BRAIN_POINTS
    if max_tumor_points is None:
        max_tumor_points = MAX_TUMOR_POINTS
        
    try:
        case_id = pred_data['case_id']
        pred = pred_data['pred']
        case_path = f"Files_Seg3D/{case_id}"
        t1ce_file = os.path.join(case_path, f"{case_id}_t1ce.nii")
        
        if not os.path.exists(t1ce_file):
            logger.error(f"T1CE file not found: {t1ce_file}")
            return None
        
        import nibabel as nib
        import plotly.graph_objects as go
        import plotly.io as pio
        
        # Load T1CE MRI scan
        t1ce_img = nib.load(t1ce_file)
        t1ce_data = t1ce_img.get_fdata()
        
        # Downsample for performance
        brain = t1ce_data[::downsample_factor, ::downsample_factor, ::downsample_factor]
        pred_down = pred[:, ::downsample_factor, ::downsample_factor, ::downsample_factor]
        brain_norm = (brain - brain.min()) / (brain.max() - brain.min())
        
        # Convert multi-channel to single-channel segmentation
        pred_seg = libs['np'].zeros_like(pred_down[0])
        pred_seg[pred_down[1] > 0.5] = 2  # ED
        pred_seg[pred_down[0] > 0.5] = 1  # NCR/NET
        pred_seg[pred_down[2] > 0.5] = 4  # ET
        
        # Extract brain tissue coordinates (threshold at 0.2)
        brain_mask = brain_norm > 0.2
        coords = libs['np'].where(brain_mask)
        if len(coords[0]) == 0:
            logger.error("No brain tissue found")
            return None
        
        # Randomly sample brain points for performance
        brain_sample_idx = libs['np'].random.choice(len(coords[0]), min(max_brain_points, len(coords[0])), replace=False)
        
        # Create 3D scatter plot
        fig = go.Figure()
        
        # Add brain tissue as background
        fig.add_trace(go.Scatter3d(
            x=coords[0][brain_sample_idx],
            y=coords[1][brain_sample_idx], 
            z=coords[2][brain_sample_idx],
            mode='markers',
            marker=dict(size=2, color='lightgray', opacity=0.4),
            name='Brain Tissue',
            showlegend=True
        ))
        
        # Add tumor regions with different colors
        pred_tumor_classes = {
            1: ("NCR/NET", "red", 0.3),
            2: ("ED", "green", 0.05), 
            4: ("ET", "blue", 0.1)
        }
        
        for lbl, (label_name, color, opacity) in pred_tumor_classes.items():
            tumor_coords = libs['np'].where(pred_seg == lbl)
            if tumor_coords[0].size > 0:
                # Sample tumor points for performance
                sample_idx = libs['np'].random.choice(len(tumor_coords[0]), min(max_tumor_points, len(tumor_coords[0])), replace=False)
                fig.add_trace(go.Scatter3d(
                    x=tumor_coords[0][sample_idx],
                    y=tumor_coords[1][sample_idx],
                    z=tumor_coords[2][sample_idx],
                    mode='markers',
                    marker=dict(size=4, color=color, opacity=opacity),
                    name=label_name,
                    showlegend=True
                ))
        
        fig.update_layout(
            title=dict(text=f'<b>3D Brain Tumor Prediction: {case_id}</b>', x=0.5, xanchor='center', font=dict(size=16)),
            scene=dict(
                xaxis_title='Sagittal', yaxis_title='Coronal', zaxis_title='Axial', 
                aspectmode='data', camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
                xaxis=dict(showbackground=True, backgroundcolor="rgb(230, 230, 230)"),
                yaxis=dict(showbackground=True, backgroundcolor="rgb(230, 230, 230)"),
                zaxis=dict(showbackground=True, backgroundcolor="rgb(230, 230, 230)")
            ),
            height=700, width=1000, margin=dict(l=0, r=0, b=80, t=120),
            legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)', bordercolor='black', borderwidth=1, font=dict(size=12))
        )
        
        # Generate HTML with centered layout
        html_content = pio.to_html(fig, include_plotlyjs='cdn')
        
        # Add CSS to center the plot in the viewport
        centered_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <style>
                body {{
                    margin: 0;
                    padding: 0;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    min-height: 100vh;
                    background-color: #f5f5f5;
                }}
                .plotly-graph-div {{
                    margin: auto;
                }}
            </style>
        </head>
        <body>
            {html_content.split('<body>')[1].split('</body>')[0]}
        </body>
        </html>
        """
        
        html_b64 = base64.b64encode(centered_html.encode()).decode()
        logger.info(f"3D prediction HTML generated for {case_id}")
        return html_b64
        
    except Exception as e:
        logger.error(f"Error creating 3D prediction HTML: {e}")
        return None

@app.post("/segment")
async def segment_brain_tumor(request: SegmentationRequest):
    """
    Run 3D brain tumor segmentation on BraTS case
    
    Expects case folder with: flair, t1, t1ce, t2, seg NIfTI files
    Returns Dice scores, visualizations, volumetric and spatial analysis
    """
    if request.model not in SEGMENTATION_MODELS:
        raise HTTPException(status_code=400, detail=f"Model {request.model} not available")
    
    if not os.path.exists(request.case_path):
        raise HTTPException(status_code=404, detail=f"Case not found: {request.case_path}")
    
    libs = load_segmentation_imports()
    case_id = os.path.basename(request.case_path)
    
    # Verify all required modalities exist
    required = ['flair', 't1', 't1ce', 't2', 'seg']
    file_paths = {}
    for mod in required:
        fp = os.path.join(request.case_path, f"{case_id}_{mod}.nii")
        if not os.path.exists(fp):
            raise HTTPException(status_code=404, detail=f"Missing: {fp}")
        file_paths[mod] = fp
    
    data_dict = {
        "image": [file_paths['flair'], file_paths['t1'], file_paths['t1ce'], file_paths['t2']],
        "label": file_paths['seg'],
        "case_id": case_id
    }
    
    transforms = get_segmentation_transforms()
    dataset = libs['Dataset'](data=[data_dict], transform=transforms)
    # Use num_workers=0 to avoid multiprocessing issues with relative imports
    dataloader = libs['DataLoader'](dataset, batch_size=1, shuffle=False, num_workers=0)
    
    model_path = SEGMENTATION_MODELS[request.model]
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"Model not found: {model_path}")
    
    session_options = libs['ort'].SessionOptions()
    ort_session = libs['ort'].InferenceSession(model_path, sess_options=session_options, 
                                                providers=['CPUExecutionProvider'])
    
    for batch_data in dataloader:
        images = batch_data["image"]
        labels = batch_data["label"]
        
        images_np = images.cpu().numpy()
        ort_inputs = {ort_session.get_inputs()[0].name: images_np}
        outputs_np = ort_session.run(None, ort_inputs)[0]
        
        outputs = libs['torch'].from_numpy(outputs_np)
        outputs = libs['torch'].sigmoid(outputs)
        outputs = (outputs > 0.5).float()
        
        dice_scores = calculate_dice_score(outputs[0], labels[0])
        
        pred_data = {
            'case_id': case_id,
            'image': images[0].cpu().numpy(),
            'pred': outputs[0].cpu().numpy(),
            'label': labels[0].cpu().numpy(),
            'dice': dice_scores
        }
        
        # Calculate volumetric analysis
        voxel_volume = 1.5 * 1.5 * 2.0  # mm³ per voxel (from spacing transform)
        pred_np = outputs[0].cpu().numpy()
        
        # Calculate volumes for each tumor region (convert mm³ to cm³)
        tc_volume = float(libs['np'].sum(pred_np[0] > 0.5) * voxel_volume / 1000)  # Tumor Core
        wt_volume = float(libs['np'].sum(pred_np[1] > 0.5) * voxel_volume / 1000)  # Whole Tumor
        et_volume = float(libs['np'].sum(pred_np[2] > 0.5) * voxel_volume / 1000)  # Enhancing Tumor
        
        # Calculate derived volumes
        ncr_volume = tc_volume - et_volume  # Necrotic core = TC - ET
        ed_volume = wt_volume - tc_volume  # Edema = WT - TC
        
        # Calculate spatial analysis: tumor location (center of mass)
        if libs['np'].sum(pred_np[1] > 0.5) > 0:  # Use whole tumor channel
            tumor_coords = libs['np'].where(pred_np[1] > 0.5)
            center_of_mass = {
                'sagittal': float(libs['np'].mean(tumor_coords[0])),
                'coronal': float(libs['np'].mean(tumor_coords[1])),
                'axial': float(libs['np'].mean(tumor_coords[2]))
            }
        else:
            center_of_mass = None
        
        # Calculate spatial analysis: tumor extent (bounding box)
        if libs['np'].sum(pred_np[1] > 0.5) > 0:
            tumor_coords = libs['np'].where(pred_np[1] > 0.5)
            extent = {
                'sagittal_range': [int(tumor_coords[0].min()), int(tumor_coords[0].max())],
                'coronal_range': [int(tumor_coords[1].min()), int(tumor_coords[1].max())],
                'axial_range': [int(tumor_coords[2].min()), int(tumor_coords[2].max())],
                'sagittal_span_mm': float((tumor_coords[0].max() - tumor_coords[0].min()) * 1.5),
                'coronal_span_mm': float((tumor_coords[1].max() - tumor_coords[1].min()) * 1.5),
                'axial_span_mm': float((tumor_coords[2].max() - tumor_coords[2].min()) * 2.0)
            }
        else:
            extent = None
        
        # Generate all visualizations
        main_viz_b64 = visualize_segmentation(pred_data, libs)
        additional_viz_b64 = generate_additional_visualizations(pred_data, libs)
        html_3d_b64 = create_3d_prediction_html(pred_data, libs)
        
        class_names = ['Tumor Core (TC)', 'Whole Tumor (WT)', 'Enhancing Tumor (ET)']
        
        result = {
            'task': 'segmentation',
            'model_used': request.model,
            'case_path': request.case_path,
            'case_id': case_id,
            'timestamp': datetime.now().isoformat(),
            'dice_scores': {class_names[i]: dice_scores[i] for i in range(len(class_names))},
            'average_dice': float(libs['np'].mean(dice_scores)),
            'volumetric_analysis': {
                'tumor_core_volume_cm3': round(tc_volume, 2),
                'whole_tumor_volume_cm3': round(wt_volume, 2),
                'enhancing_tumor_volume_cm3': round(et_volume, 2),
                'necrotic_core_volume_cm3': round(ncr_volume, 2),
                'edema_volume_cm3': round(ed_volume, 2),
                'voxel_spacing_mm': [1.5, 1.5, 2.0]
            },
            'spatial_analysis': {
                'center_of_mass': center_of_mass,
                'tumor_extent': extent
            },
            'visualization': main_viz_b64,
            'additional_visualizations': additional_viz_b64,
            'class_names': class_names
        }
        
        if html_3d_b64:
            result['3d_html_visualization'] = html_3d_b64
        
        return result

@app.get("/models")
async def get_models():
    available_class = [m for m, p in CLASSIFICATION_MODELS.items() if os.path.exists(p)]
    available_seg = [m for m, p in SEGMENTATION_MODELS.items() if os.path.exists(p)]
    return {
        'detection_models': list(detection_models.keys()),
        'classification_models': available_class,
        'segmentation_models': available_seg
    }

@app.get("/health")
async def health_check():
    available_class = [m for m, p in CLASSIFICATION_MODELS.items() if os.path.exists(p)]
    available_seg = [m for m, p in SEGMENTATION_MODELS.items() if os.path.exists(p)]
    return {
        'status': 'healthy',
        'detection_models': len(detection_models),
        'classification_models': len(available_class),
        'segmentation_models': len(available_seg)
    }

@app.get("/")
async def root():
    return {
        'message': 'Medical AI Models Server',
        'version': '3.0',
        'endpoints': {
            'detection': '/detect',
            'classification': '/classify',
            'segmentation': '/segment',
            'models': '/models',
            'health': '/health'
        }
    }

@app.on_event("startup")
async def startup_event():
    logger.info("="*80)
    logger.info("MEDICAL AI MODELS SERVER")
    logger.info("="*80)
    
    load_detection_models()
    load_all_classification_models()
    
    available_segmentation = []
    for model_name, model_path in SEGMENTATION_MODELS.items():
        if os.path.exists(model_path):
            available_segmentation.append(model_name)
    logger.info(f"Segmentation models available: {len(available_segmentation)}")
    logger.info(f"Available: {available_segmentation}")
    
    logger.info("="*80)
    logger.info("SERVER READY - ALL MODELS PRE-LOADED!")
    logger.info("="*80)

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)

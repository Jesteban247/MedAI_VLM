# ============================================================================
# IMPORTS AND DEPENDENCIES
# ============================================================================

import os
import time
import json
import torch
import logging
import warnings
import argparse
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from PIL import Image
from tqdm.auto import tqdm
from datetime import datetime
from accelerate import Accelerator
from accelerate.utils import set_seed
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset, DataLoader
from transformers import logging as transformers_logging
from torchvision.transforms import Compose, Normalize, Resize, CenterCrop, ToTensor
from transformers import AutoConfig, AutoImageProcessor, AutoModelForImageClassification
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, roc_auc_score, confusion_matrix

# ============================================================================
# CONFIGURATION AND CONSTANTS
# ============================================================================

# Use 'Agg' backend for matplotlib to avoid GUI issues
matplotlib.use('Agg')

# Suppress warnings
warnings.filterwarnings("ignore")
transformers_logging.set_verbosity_error()

# Supported models with LoRA configurations
MODEL_CONFIGS = {
    "microsoft/resnet-18": {"lora_target_modules": ["convolution"]},
    "google/efficientnet-b1": {"lora_target_modules": ["convolution"]},
    "facebook/deit-tiny-patch16-224": {"lora_target_modules": ["query", "value"]},
    "microsoft/swin-tiny-patch4-window7-224": {"lora_target_modules": ["query", "value"]},
    "google/vit-base-patch16-384": {"lora_target_modules": ["query", "value"]},
    "microsoft/cvt-13": {"lora_target_modules": ["projection_query", "projection_value"]},
    "apple/mobilevit-small":{"lora_target_modules": ["query", "value"]},
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_image_size(processor):
    """Extract image size from processor consistently."""
    if hasattr(processor, 'size'):
        if isinstance(processor.size, dict):
            return processor.size.get("shortest_edge", processor.size.get("height", 224))
        return processor.size
    return 224

def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def format_time(seconds):
    """Format seconds into readable time string."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        return f"{seconds/60:.2f}m"
    else:
        return f"{seconds/3600:.2f}h"


# ============================================================================
# DATASET AND DATALOADER
# ============================================================================

def get_class_names(data_dir):
    """Get sorted list of class names from directory structure."""
    train_dir = os.path.join(data_dir, 'train')
    if not os.path.exists(train_dir):
        raise ValueError(f"Train directory not found: {train_dir}")
    classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    if not classes:
        raise ValueError(f"No class directories found in: {train_dir}")
    return classes

class ImageDataset(Dataset):
    """Custom dataset for image classification."""
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        img_path, label = self.file_list[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

def collect_files(data_dir, class_names):
    """Collect image files from train/val/test splits."""
    splits = {'train': [], 'val': [], 'test': []}
    label_map = {name: idx for idx, name in enumerate(class_names)}
    
    for split in splits.keys():
        split_dir = os.path.join(data_dir, split)
        if not os.path.exists(split_dir):
            continue
            
        for class_name in class_names:
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.exists(class_dir):
                continue
                
            for f in os.listdir(class_dir):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, f)
                    splits[split].append((img_path, label_map[class_name]))
    
    return splits['train'], splits['val'], splits['test']

def create_data_loaders(args, processor, class_names):
    """Create DataLoaders for train, val, and test splits."""
    train_files, val_files, test_files = collect_files(args.data_dir, class_names)
    
    if not train_files:
        raise ValueError("No training images found")
    
    size = get_image_size(processor)
    if hasattr(processor, 'image_mean') and hasattr(processor, 'image_std'):
        normalize = Normalize(mean=processor.image_mean, std=processor.image_std)
    else:
        normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transforms = Compose([Resize(size), CenterCrop(size), ToTensor(), normalize])

    train_dataset = ImageDataset(train_files, transform=transforms)
    val_dataset = ImageDataset(val_files, transform=transforms) if val_files else None
    test_dataset = ImageDataset(test_files, transform=transforms) if test_files else None
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, persistent_workers=True
    ) if val_dataset else None
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, persistent_workers=True
    ) if test_dataset else None
    
    return train_loader, val_loader, test_loader, len(train_files), len(val_files), len(test_files)


# ============================================================================
# MODEL SETUP
# ============================================================================

def save_base_model(args, processor, accelerator, logger):
    """Save the base pretrained model before applying LoRA."""
    if not args.use_lora or not accelerator.is_main_process or args.fast_mode:
        return
    
    base_dir = os.path.join(args.output_dir, 'base_model')
    os.makedirs(base_dir, exist_ok=True)
    
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    model = AutoModelForImageClassification.from_pretrained(args.model_name_or_path, config=config)
    
    model.save_pretrained(base_dir)
    processor.save_pretrained(base_dir)
    
    logger.info(f"Base model saved to: {base_dir}")

def create_model(args, num_classes, id2label, label2id):
    """Create model with optional LoRA."""
    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        num_labels=num_classes,
        id2label=id2label,
        label2id=label2id
    )
    
    model = AutoModelForImageClassification.from_pretrained(
        args.model_name_or_path,
        config=config,
        ignore_mismatched_sizes=True
    )
    
    if args.use_lora:
        target_modules = MODEL_CONFIGS.get(args.model_name_or_path, {}).get("lora_target_modules")
        if not target_modules:
            raise ValueError(f"Model {args.model_name_or_path} not supported for LoRA. "
                           f"Supported models: {list(MODEL_CONFIGS.keys())}")
        
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=args.lora_dropout,
            bias="none",
            modules_to_save=["classifier"]
        )
        model = get_peft_model(model, lora_config)
    
    return model


# ============================================================================
# METRICS AND EVALUATION
# ============================================================================

def calculate_metrics(preds, targets, probs, num_classes):
    """Calculate per-class metrics."""
    preds_np = preds.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()
    probs_np = probs.detach().cpu().numpy()
    
    metrics = {'precision': [], 'recall': [], 'accuracy': [], 'f1': [], 'auc': []}
    
    for class_idx in range(num_classes):
        target_bin = (targets_np == class_idx).astype(int)
        pred_bin = (preds_np == class_idx).astype(int)
        
        metrics['precision'].append(precision_score(target_bin, pred_bin, zero_division=0))
        metrics['recall'].append(recall_score(target_bin, pred_bin, zero_division=0))
        metrics['accuracy'].append(accuracy_score(target_bin, pred_bin))
        metrics['f1'].append(f1_score(target_bin, pred_bin, zero_division=0))
        metrics['auc'].append(
            roc_auc_score(target_bin, probs_np[:, class_idx]) if target_bin.sum() > 0 else 0.0
        )
    
    return metrics

def run_epoch(model, loader, criterion, optimizer, accelerator, num_classes, is_train=True, desc=""):
    """Run one epoch of training or evaluation."""
    if not loader:
        return None, None, None, None, None
    
    start_time = time.time()
    model.train() if is_train else model.eval()
    
    total_loss = 0
    all_preds, all_targets, all_probs = [], [], []
    
    pbar = tqdm(loader, desc=desc, disable=not accelerator.is_local_main_process)
    
    context = torch.no_grad() if not is_train else torch.enable_grad()
    
    with context:
        for inputs, targets in pbar:
            with accelerator.autocast():
                outputs = model(inputs).logits
                loss = criterion(outputs, targets)
            
            if is_train:
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
            
            total_loss += loss.item()
            probs = torch.softmax(outputs, dim=-1)
            preds = torch.argmax(probs, dim=-1)
            
            all_preds.append(preds)
            all_targets.append(targets)
            all_probs.append(probs)
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    all_preds = accelerator.gather_for_metrics(torch.cat(all_preds))
    all_targets = accelerator.gather_for_metrics(torch.cat(all_targets))
    all_probs = accelerator.gather_for_metrics(torch.cat(all_probs))
    
    avg_loss = total_loss / len(loader)
    metrics = calculate_metrics(all_preds, all_targets, all_probs, num_classes)
    epoch_time = time.time() - start_time
    
    return avg_loss, metrics, all_preds.cpu().numpy(), all_targets.cpu().numpy(), epoch_time


# ============================================================================
# PLOTTING AND SAVING
# ============================================================================

def plot_confusion_matrix(preds, targets, class_names, output_dir, phase):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(targets, preds)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {phase.capitalize()}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'confusion_matrix_{phase}.png'), dpi=150)
    plt.close()

def plot_training_curves(history, class_names, output_dir):
    """Plot training and validation curves."""
    epochs = range(1, len(history['train_loss']) + 1)
    has_val = 'val_loss' in history and len(history['val_loss']) > 0
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    axes[0, 0].plot(epochs, history['train_loss'], 'b-o', label='Train')
    if has_val:
        axes[0, 0].plot(epochs, history['val_loss'], 'r--s', label='Val')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    for idx, metric in enumerate(['accuracy', 'precision', 'recall', 'f1', 'auc']):
        ax = axes[(idx + 1) // 3, (idx + 1) % 3]
        
        train_avg = [np.mean(m[metric]) for m in history['train_metrics']]
        ax.plot(epochs, train_avg, 'b-o', label='Train')
        
        if has_val:
            val_avg = [np.mean(m[metric]) for m in history['val_metrics']]
            ax.plot(epochs, val_avg, 'r--s', label='Val')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.capitalize())
        ax.set_title(metric.capitalize())
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    for idx, metric in enumerate(['accuracy', 'f1', 'precision', 'recall']):
        ax = axes[idx // 2, idx % 2]
        
        for class_idx, class_name in enumerate(class_names):
            train_vals = [m[metric][class_idx] for m in history['train_metrics']]
            ax.plot(epochs, train_vals, '-o', label=f'{class_name} (Train)', alpha=0.7)
            
            if has_val:
                val_vals = [m[metric][class_idx] for m in history['val_metrics']]
                ax.plot(epochs, val_vals, '--s', label=f'{class_name} (Val)', alpha=0.7)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f'{metric.capitalize()} per Class')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'per_class_metrics.png'), dpi=150, bbox_inches='tight')
    plt.close()

def save_results(history, class_names, output_dir):
    """Save results to CSV files."""
    data = []
    
    for epoch in range(len(history['train_loss'])):
        row = {
            'epoch': epoch + 1,
            'phase': 'train',
            'loss': history['train_loss'][epoch],
            'time_seconds': history['train_time'][epoch]
        }
        metrics = history['train_metrics'][epoch]
        for metric_name in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
            row[f'avg_{metric_name}'] = np.mean(metrics[metric_name])
            for class_idx, class_name in enumerate(class_names):
                row[f'{class_name}_{metric_name}'] = metrics[metric_name][class_idx]
        data.append(row)
    
    if 'val_loss' in history and len(history['val_loss']) > 0:
        for epoch in range(len(history['val_loss'])):
            row = {
                'epoch': epoch + 1,
                'phase': 'val',
                'loss': history['val_loss'][epoch],
                'time_seconds': history['val_time'][epoch]
            }
            metrics = history['val_metrics'][epoch]
            for metric_name in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
                row[f'avg_{metric_name}'] = np.mean(metrics[metric_name])
                for class_idx, class_name in enumerate(class_names):
                    row[f'{class_name}_{metric_name}'] = metrics[metric_name][class_idx]
            data.append(row)
    
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(output_dir, 'training_history.csv'), index=False)
    
    if 'test_loss' in history:
        test_row = {
            'phase': 'test',
            'loss': history['test_loss'],
            'time_seconds': history.get('test_time', 0)
        }
        metrics = history['test_metrics']
        for metric_name in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
            test_row[f'avg_{metric_name}'] = np.mean(metrics[metric_name])
            for class_idx, class_name in enumerate(class_names):
                test_row[f'{class_name}_{metric_name}'] = metrics[metric_name][class_idx]
        
        test_df = pd.DataFrame([test_row])
        test_df.to_csv(os.path.join(output_dir, 'test_results.csv'), index=False)

def save_config(args, class_names, output_dir, total_params, trainable_params):
    """Save training configuration."""
    config = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'data_dir': args.data_dir,
        'model': args.model_name_or_path,
        'classes': class_names,
        'num_classes': len(class_names),
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.lr,
        'optimizer': args.optimizer,
        'use_lora': args.use_lora,
        'model_parameters': {
            'total': total_params,
            'trainable': trainable_params,
            'frozen': total_params - trainable_params,
            'trainable_percentage': round(100 * trainable_params / total_params, 2)
        }
    }
    
    if args.use_lora:
        config['lora_config'] = {
            'r': args.lora_r,
            'alpha': args.lora_alpha,
            'dropout': args.lora_dropout,
            'target_modules': MODEL_CONFIGS[args.model_name_or_path]['lora_target_modules']
        }
    
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Image Classification Training")
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--model_name_or_path', type=str, default='microsoft/resnet-18',
                        choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay for optimizer')
    parser.add_argument('--optimizer', type=str, default='AdamW', choices=['Adam', 'AdamW', 'SGD'])
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--use_lora', action='store_true')
    parser.add_argument('--lora_r', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--lora_dropout', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--fast_mode', action='store_true', help='Skip plotting and saving for faster execution')
    parser.add_argument('--save_dir', type=str, default=None, help='Custom directory to save all outputs')
    args = parser.parse_args()
    
    accelerator = Accelerator(mixed_precision='fp16')
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_name = args.model_name_or_path.split('/')[-1]
    lora_suffix = '_lora' if args.use_lora else ''
    
    if args.save_dir:
        args.output_dir = os.path.join(args.output_dir, args.save_dir)
    else:
        args.output_dir = os.path.join(args.output_dir, f'{model_name}{lora_suffix}_{timestamp}')
    
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(os.path.join(args.output_dir, 'training.log'))
            ],
            force=True
        )
    
    logger = logging.getLogger(__name__)
    
    if args.seed is not None:
        set_seed(args.seed)
    
    class_names = get_class_names(args.data_dir)
    num_classes = len(class_names)
    id2label = {i: name for i, name in enumerate(class_names)}
    label2id = {name: i for i, name in enumerate(class_names)}
    
    if accelerator.is_main_process:
        logger.info("="*80)
        logger.info(f"Training Configuration")
        logger.info("="*80)
        logger.info(f"Model: {args.model_name_or_path}")
        logger.info(f"Classes ({num_classes}): {', '.join(class_names)}")
        logger.info(f"GPUs: {accelerator.num_processes}")
        logger.info(f"Batch size per GPU: {args.batch_size}")
        logger.info(f"Effective batch size: {args.batch_size * accelerator.num_processes}")
        logger.info(f"Epochs: {args.epochs}")
        logger.info(f"Learning rate: {args.lr}")
        logger.info(f"Optimizer: {args.optimizer}")
        logger.info(f"Use LoRA: {args.use_lora}")
        if args.use_lora:
            logger.info(f"  LoRA rank: {args.lora_r}")
            logger.info(f"  LoRA alpha: {args.lora_alpha}")
            logger.info(f"  LoRA dropout: {args.lora_dropout}")
        logger.info(f"Fast mode (no plots): {args.fast_mode}")
        logger.info("="*80)
    
    processor = AutoImageProcessor.from_pretrained(args.model_name_or_path)
    
    save_base_model(args, processor, accelerator, logger)
    
    train_loader, val_loader, test_loader, n_train, n_val, n_test = create_data_loaders(
        args, processor, class_names
    )
    
    if accelerator.is_main_process:
        logger.info(f"Dataset: Train={n_train}, Val={n_val}, Test={n_test}")

    model = create_model(args, num_classes, id2label, label2id)

    total_params, trainable_params = count_parameters(model)
    trainable_pct = 100 * trainable_params / total_params
    
    if accelerator.is_main_process:
        logger.info("="*80)
        logger.info(f"Model Parameters:")
        logger.info(f"  Total:      {total_params:,}")
        logger.info(f"  Trainable:  {trainable_params:,} ({trainable_pct:.2f}%)")
        logger.info(f"  Frozen:     {total_params - trainable_params:,}")
        logger.info("="*80)
        
        save_config(args, class_names, args.output_dir, total_params, trainable_params)
    
    if args.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    criterion = nn.CrossEntropyLoss()
    
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    if val_loader:
        val_loader = accelerator.prepare(val_loader)
    if test_loader:
        test_loader = accelerator.prepare(test_loader)
    
    history = {
        'train_loss': [], 'train_metrics': [], 'train_time': [],
        'val_loss': [], 'val_metrics': [], 'val_time': []
    }
    
    if accelerator.is_main_process:
        logger.info("\nStarting training...")
        total_start = time.time()
    
    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics, train_preds, train_targets, train_time = run_epoch(
            model, train_loader, criterion, optimizer, accelerator, num_classes,
            is_train=True, desc=f"Train [{epoch}/{args.epochs}]"
        )
        
        history['train_loss'].append(train_loss)
        history['train_metrics'].append(train_metrics)
        history['train_time'].append(train_time)
        
        if val_loader:
            val_loss, val_metrics, val_preds, val_targets, val_time = run_epoch(
                model, val_loader, criterion, None, accelerator, num_classes,
                is_train=False, desc=f"Val   [{epoch}/{args.epochs}]"
            )
            history['val_loss'].append(val_loss)
            history['val_metrics'].append(val_metrics)
            history['val_time'].append(val_time)
        
        if accelerator.is_main_process:
            log_msg = f"Epoch {epoch}/{args.epochs} | "
            log_msg += f"Train Loss: {train_loss:.4f}, Acc: {np.mean(train_metrics['accuracy']):.4f}, "
            log_msg += f"F1: {np.mean(train_metrics['f1']):.4f}, Time: {format_time(train_time)}"
            
            if val_loader:
                log_msg += f" | Val Loss: {val_loss:.4f}, Acc: {np.mean(val_metrics['accuracy']):.4f}, "
                log_msg += f"F1: {np.mean(val_metrics['f1']):.4f}, Time: {format_time(val_time)}"
            
            logger.info(log_msg)
    
    if accelerator.is_main_process:
        total_time = time.time() - total_start
        logger.info(f"\nTotal training time: {format_time(total_time)}")
    
    if test_loader:
        test_loss, test_metrics, test_preds, test_targets, test_time = run_epoch(
            model, test_loader, criterion, None, accelerator, num_classes,
            is_train=False, desc="Test"
        )
        
        if accelerator.is_main_process:
            history['test_loss'] = test_loss
            history['test_metrics'] = test_metrics
            history['test_time'] = test_time
            
            logger.info("="*80)
            logger.info(f"Test Results:")
            logger.info(f"  Loss: {test_loss:.4f}")
            logger.info(f"  Accuracy: {np.mean(test_metrics['accuracy']):.4f}")
            logger.info(f"  F1 Score: {np.mean(test_metrics['f1']):.4f}")
            logger.info(f"  Time: {format_time(test_time)}")
            logger.info("="*80)
    
    if accelerator.is_main_process:
        if not args.fast_mode:
            unwrapped = accelerator.unwrap_model(model)
            
            if args.use_lora:
                lora_adapter_dir = os.path.join(args.output_dir, 'lora_adapter')
                os.makedirs(lora_adapter_dir, exist_ok=True)
                unwrapped.save_pretrained(lora_adapter_dir)
                processor.save_pretrained(lora_adapter_dir)
                logger.info(f"LoRA adapter saved to: {lora_adapter_dir}")

                model_dir = os.path.join(args.output_dir, 'model')
                os.makedirs(model_dir, exist_ok=True)
                logger.info("Merging LoRA weights with base model...")
                merged_model = unwrapped.merge_and_unload()
                merged_model.save_pretrained(model_dir)
                processor.save_pretrained(model_dir)
                logger.info(f"Merged model saved to: {model_dir}")
            else:
                model_dir = os.path.join(args.output_dir, 'model')
                os.makedirs(model_dir, exist_ok=True)
                unwrapped.save_pretrained(model_dir)
                processor.save_pretrained(model_dir)
                logger.info(f"\nModel saved to: {model_dir}")
        else:
            logger.info("\nFast mode enabled - skipping model saving.")
        
        if not args.fast_mode:
            logger.info("Generating plots...")
            plot_confusion_matrix(train_preds, train_targets, class_names, args.output_dir, 'train')
            
            if val_loader:
                plot_confusion_matrix(val_preds, val_targets, class_names, args.output_dir, 'val')
            if test_loader:
                plot_confusion_matrix(test_preds, test_targets, class_names, args.output_dir, 'test')

            plot_training_curves(history, class_names, args.output_dir)
            logger.info("Plots saved successfully.")
        else:
            logger.info("Fast mode enabled - skipping plot generation.")
        
        save_results(history, class_names, args.output_dir)
        
        logger.info(f"Results saved to: {args.output_dir}")
    
    accelerator.end_training()

if __name__ == '__main__':
    main()
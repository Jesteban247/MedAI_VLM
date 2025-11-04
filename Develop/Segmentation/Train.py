# ============================================================================
# IMPORTS AND DEPENDENCIES
# ============================================================================

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import logging
import json
import csv
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, accuracy_score
import random
import argparse
from monai.data import Dataset, DataLoader as MonaiDataLoader, CacheDataset
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, EnsureTyped, Orientationd, Spacingd,
    NormalizeIntensityd, Activations, AsDiscrete, MapTransform, CropForegroundd, SpatialPadd
)
from monai.networks.nets import SegResNet
from monai.losses import DiceFocalLoss
from monai.inferers import sliding_window_inference
from monai.utils import set_determinism
import time
from accelerate import Accelerator
from accelerate.utils import set_seed

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def setup_logging(output_dir, accelerator):
    """Setup logging to file and console (only on main process)"""
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        log_file = os.path.join(output_dir, 'training.log')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    else:
        logging.basicConfig(level=logging.WARNING)
    return logging.getLogger()

def save_config(args, output_dir):
    """Save configuration to text file"""
    config_path = os.path.join(output_dir, 'config.txt')
    with open(config_path, 'w') as f:
        f.write("BraTS 3D Segmentation Training Configuration with MONAI + Accelerate\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("Data Configuration:\n")
        f.write(f"  Data Directory: {args.data_dir}\n")
        f.write(f"  ROI Size (Inference): {args.roi_size}\n")
        f.write(f"  Number of Classes: {args.num_classes}\n")
        f.write(f"  Train Split: {args.train_split * 100:.1f}%\n")
        f.write(f"  Val Split: {args.val_split * 100:.1f}%\n")
        f.write(f"  Test Split: {args.test_split * 100:.1f}%\n")
        f.write(f"  No Augmentation (Original Images Only)\n\n")
        
        f.write("Model Configuration:\n")
        f.write(f"  Model: SegResNet\n")
        f.write(f"  Input Channels: {args.in_channels}\n")
        f.write(f"  Output Channels: {args.num_classes}\n")
        f.write(f"  Initial Filters: {args.init_filters}\n\n")
        
        f.write("Training Configuration:\n")
        f.write(f"  Framework: Accelerate\n")
        f.write(f"  Mixed Precision: {args.mixed_precision}\n")
        f.write(f"  Optimizer: {args.optimizer}\n")
        f.write(f"  Batch Size (per GPU): {args.batch_size}\n")
        f.write(f"  Batch Size (Val/Test per GPU): {args.val_batch_size}\n")
        f.write(f"  Epochs: {args.epochs}\n")
        f.write(f"  Learning Rate: {args.lr}\n")
        f.write(f"  Weight Decay: {args.weight_decay}\n")
        if args.optimizer == 'SGD':
            f.write(f"  Momentum: {args.momentum}\n")
        f.write(f"  Validation Interval: {args.val_interval}\n")
        f.write(f"  Use Caching: {args.use_cache}\n\n")
        
        f.write("Hardware Configuration:\n")
        f.write(f"  Number of Workers per GPU: {args.num_workers}\n")
        f.write(f"  Pin Memory: {args.pin_memory}\n")
        f.write(f"  Prefetch Factor: {args.prefetch_factor}\n\n")
        
        f.write("Class Names (Multi-Label):\n")
        for i, name in enumerate(args.class_names):
            f.write(f"  Channel {i}: {name}\n")
    
    logging.getLogger().info(f"Configuration saved: {config_path}")

def get_data_list(data_dir):
    """Get list of data dictionaries for MONAI"""
    cases = sorted([d for d in os.listdir(data_dir) 
                    if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('BraTS20_Training_')])
    
    data_list = []
    for case in cases:
        base_path = os.path.join(data_dir, case)
        img_paths = [
            os.path.join(base_path, f'{case}_flair.nii'),
            os.path.join(base_path, f'{case}_t1.nii'),
            os.path.join(base_path, f'{case}_t1ce.nii'),
            os.path.join(base_path, f'{case}_t2.nii')
        ]
        label_path = os.path.join(base_path, f'{case}_seg.nii')
        data_list.append({
            "image": img_paths,
            "label": label_path,
            "case_id": case
        })
    return data_list

class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    """Convert single-channel labels to multi-channel based on BraTS 2020 regions"""
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            result.append(torch.logical_or(d[key] == 1, d[key] == 4))
            result.append(torch.logical_or(torch.logical_or(d[key] == 1, d[key] == 4), d[key] == 2))
            result.append(d[key] == 4)
            d[key] = torch.stack(result, axis=0).float()
        return d

def get_transforms(args):
    """Get transforms WITHOUT augmentation - original images only"""
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys="image"),
        EnsureTyped(keys=["image", "label"]),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        # Pad to a common size to enable batching (all images will be same size)
        SpatialPadd(keys=["image", "label"], spatial_size=args.roi_size, mode="constant"),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ])

# ============================================================================
# METRICS AND EVALUATION
# ============================================================================

def calculate_comprehensive_metrics(pred, target, num_classes=3, epsilon=1e-8):
    """Calculate Precision, Recall, Accuracy, and Dice for each channel"""
    metrics = {'precision': [], 'recall': [], 'accuracy': [], 'dice': []}
    
    B, C, *spatial = pred.shape
    assert C == num_classes
    
    for c in range(num_classes):
        pred_c = pred[:, c].reshape(-1).cpu().numpy()
        target_c = target[:, c].reshape(-1).cpu().numpy()
        
        pred_binary = (pred_c > 0.5).astype(int)
        target_binary = (target_c > 0.5).astype(int)
        
        if target_binary.sum() > 0:
            precision = precision_score(target_binary, pred_binary, zero_division=0)
            recall = recall_score(target_binary, pred_binary, zero_division=0)
            accuracy = accuracy_score(target_binary, pred_binary)
        else:
            precision = recall = accuracy = 0.0
        
        pred_class = pred[:, c]
        target_class = target[:, c]
        intersection = (pred_class * target_class).sum()
        union = pred_class.sum() + target_class.sum()
        dice = (2.0 * intersection + epsilon) / (union + epsilon)
        
        metrics['precision'].append(precision)
        metrics['recall'].append(recall)
        metrics['accuracy'].append(accuracy)
        metrics['dice'].append(dice.item())
    
    return metrics

def init_csv_logger(output_dir, phase, class_names):
    """Initialize CSV file for logging metrics"""
    csv_path = os.path.join(output_dir, f'{phase}_metrics.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['epoch', 'loss', 'avg_dice', 'avg_precision', 'avg_recall', 'avg_accuracy']
        for cls in class_names:
            header.extend([f'{cls}_dice', f'{cls}_precision', f'{cls}_recall', f'{cls}_accuracy'])
        header.append('epoch_time_sec')
        writer.writerow(header)
    return csv_path

def log_to_csv(csv_path, epoch, loss, metrics, epoch_time, class_names):
    """Log metrics to CSV file"""
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        row = [
            epoch,
            f"{loss:.6f}",
            f"{np.mean(metrics['dice']):.6f}",
            f"{np.mean(metrics['precision']):.6f}",
            f"{np.mean(metrics['recall']):.6f}",
            f"{np.mean(metrics['accuracy']):.6f}"
        ]
        for i in range(len(class_names)):
            row.extend([
                f"{metrics['dice'][i]:.6f}",
                f"{metrics['precision'][i]:.6f}",
                f"{metrics['recall'][i]:.6f}",
                f"{metrics['accuracy'][i]:.6f}"
            ])
        row.append(f"{epoch_time:.2f}")
        writer.writerow(row)

def print_metrics_table(metrics, class_names, epoch, total_epochs, phase='TRAIN'):
    """Print metrics in a formatted table"""
    print(f"\n{'='*75}")
    print(f"EPOCH {epoch}/{total_epochs} - {phase} METRICS")
    print(f"{'='*75}")
    print(f"{'Class':<20} ║ {'Precision':<9} ║ {'Recall':<9} ║ {'Accuracy':<9} ║ {'Dice':<9}")
    print(f"{'='*75}")
    
    for i, class_name in enumerate(class_names):
        print(f"{class_name:<20} ║ "
              f"{metrics['precision'][i]:<9.4f} ║ "
              f"{metrics['recall'][i]:<9.4f} ║ "
              f"{metrics['accuracy'][i]:<9.4f} ║ "
              f"{metrics['dice'][i]:<9.4f}")
    
    print(f"{'-'*75}")
    print(f"{'AVERAGE':<20} ║ "
          f"{np.mean(metrics['precision']):<9.4f} ║ "
          f"{np.mean(metrics['recall']):<9.4f} ║ "
          f"{np.mean(metrics['accuracy']):<9.4f} ║ "
          f"{np.mean(metrics['dice']):<9.4f}")
    print(f"{'='*75}\n")

# ============================================================================
# DATASET AND DATALOADER
# ============================================================================

def create_data_loaders(args, accelerator, output_dir):
    """Create train/val/test data loaders with proper 80/5/15 split"""
    logger = logging.getLogger()
    if accelerator.is_main_process:
        logger.info(f"Loading data from: {args.data_dir}")
    
    data_list = get_data_list(args.data_dir)
    
    if accelerator.is_main_process:
        logger.info(f"Found {len(data_list)} cases")
    
    if len(data_list) < 10:
        raise ValueError(f"Insufficient data. Found only {len(data_list)} samples.")
    
    # Shuffle with fixed seed
    random.seed(args.seed)
    random.shuffle(data_list)
    
    # Calculate split indices
    total = len(data_list)
    train_end = int(total * args.train_split)
    val_end = train_end + int(total * args.val_split)
    
    train_data_list = data_list[:train_end]
    val_data_list = data_list[train_end:val_end]
    test_data_list = data_list[val_end:]
    
    if accelerator.is_main_process:
        logger.info(f"\nData Split Distribution:")
        logger.info(f"  Total samples: {total}")
        logger.info(f"  Train: {len(train_data_list)} ({len(train_data_list)/total*100:.1f}%)")
        logger.info(f"  Val: {len(val_data_list)} ({len(val_data_list)/total*100:.1f}%)")
        logger.info(f"  Test: {len(test_data_list)} ({len(test_data_list)/total*100:.1f}%)")
        logger.info(f"  Seed: {args.seed}\n")
        
        # Save data split to JSON
        split_data = {
            "train": [item["case_id"] for item in train_data_list],
            "val": [item["case_id"] for item in val_data_list],
            "test": [item["case_id"] for item in test_data_list]
        }
        split_path = os.path.join(output_dir, 'data_split.json')
        with open(split_path, 'w') as f:
            json.dump(split_data, f, indent=4)
        logger.info(f"Data split saved: {split_path}")
    
    # Same transforms for all (no augmentation)
    transforms = get_transforms(args)
    
    # Use CacheDataset for faster loading if requested
    DatasetClass = CacheDataset if args.use_cache else Dataset
    
    train_dataset = DatasetClass(data=train_data_list, transform=transforms)
    val_dataset = DatasetClass(data=val_data_list, transform=transforms) if val_data_list else None
    test_dataset = DatasetClass(data=test_data_list, transform=transforms) if test_data_list else None
    
    train_loader = MonaiDataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers, drop_last=True,
        prefetch_factor=args.prefetch_factor
    )
    
    val_loader = MonaiDataLoader(
        val_dataset, batch_size=args.val_batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers, drop_last=False,
        prefetch_factor=args.prefetch_factor
    ) if val_dataset else None
    
    test_loader = MonaiDataLoader(
        test_dataset, batch_size=args.val_batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers, drop_last=False,
        prefetch_factor=args.prefetch_factor
    ) if test_dataset else None
    
    return train_loader, val_loader, test_loader

# ============================================================================
# MODEL SETUP
# ============================================================================

def create_model(args):
    """Create the segmentation model"""
    model = SegResNet(
        blocks_down=[1, 2, 2, 4],
        blocks_up=[1, 1, 1],
        init_filters=args.init_filters,
        in_channels=args.in_channels,
        out_channels=args.num_classes,
        dropout_prob=0.2,
    )
    
    return model

# ============================================================================
# TRAINING AND EVALUATION
# ============================================================================

post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

def train_epoch(model, train_loader, criterion, optimizer, accelerator, args):
    """Train one epoch"""
    model.train()
    total_loss = 0.0
    all_metrics = {k: [0.0] * args.num_classes for k in ['precision', 'recall', 'accuracy', 'dice']}
    num_batches = len(train_loader)
    
    iterator = tqdm(train_loader, desc="Train") if accelerator.is_main_process else train_loader
    
    for batch_data in iterator:
        inputs = batch_data["image"]
        labels = batch_data["label"]
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        accelerator.backward(loss)
        optimizer.step()
        
        total_loss += loss.item()
        
        with torch.no_grad():
            pred = post_trans(outputs).float()
            batch_metrics = calculate_comprehensive_metrics(pred, labels, args.num_classes)
            for k in all_metrics.keys():
                for i in range(args.num_classes):
                    all_metrics[k][i] += batch_metrics[k][i]
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    for k in all_metrics.keys():
        all_metrics[k] = [s / num_batches if num_batches > 0 else 0.0 for s in all_metrics[k]]
    
    # Gather metrics across all processes
    avg_loss = accelerator.gather(torch.tensor(avg_loss).to(accelerator.device)).mean().item()
    for k in all_metrics.keys():
        for i in range(args.num_classes):
            metric_tensor = torch.tensor(all_metrics[k][i]).to(accelerator.device)
            all_metrics[k][i] = accelerator.gather(metric_tensor).mean().item()
    
    return avg_loss, all_metrics

def eval_epoch(model, loader, criterion, accelerator, args):
    """Evaluate one epoch"""
    if loader is None:
        return 0.0, None
    
    model.eval()
    total_loss = 0.0
    all_metrics = {k: [0.0] * args.num_classes for k in ['precision', 'recall', 'accuracy', 'dice']}
    num_batches = len(loader)
    
    iterator = tqdm(loader, desc="Eval") if accelerator.is_main_process else loader
    
    with torch.no_grad():
        for batch_data in iterator:
            inputs = batch_data["image"]
            labels = batch_data["label"]
            
            outputs = sliding_window_inference(
                inputs, roi_size=args.roi_size, sw_batch_size=args.sw_batch_size,
                predictor=model, overlap=0.5, mode='gaussian'
            )
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            pred = post_trans(outputs).float()
            batch_metrics = calculate_comprehensive_metrics(pred, labels, args.num_classes)
            
            for k in all_metrics.keys():
                for i in range(args.num_classes):
                    all_metrics[k][i] += batch_metrics[k][i]
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    for k in all_metrics.keys():
        all_metrics[k] = [s / num_batches if num_batches > 0 else 0.0 for s in all_metrics[k]]
    
    # Gather metrics across all processes
    avg_loss = accelerator.gather(torch.tensor(avg_loss).to(accelerator.device)).mean().item()
    for k in all_metrics.keys():
        for i in range(args.num_classes):
            metric_tensor = torch.tensor(all_metrics[k][i]).to(accelerator.device)
            all_metrics[k][i] = accelerator.gather(metric_tensor).mean().item()
    
    return avg_loss, all_metrics

# ============================================================================
# PLOTTING AND SAVING
# ============================================================================

def save_checkpoint(model, optimizer, epoch, output_dir, args, accelerator, is_final=False):
    """Save model checkpoint"""
    if not accelerator.is_main_process:
        return None
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    name = f'model_final_{timestamp}.pth' if is_final else f'model_epoch_{epoch}_{timestamp}.pth'
    path = os.path.join(output_dir, name)
    
    # Get unwrapped model
    unwrapped_model = accelerator.unwrap_model(model)
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': unwrapped_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'args': vars(args),
    }, path)
    
    logging.getLogger().info(f"Checkpoint saved: {path}")
    return path

def plot_training_curves(train_csv, val_csv, output_dir):
    """Plot training curves from CSV files"""
    train_df = pd.read_csv(train_csv)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Loss
    axes[0, 0].plot(train_df['epoch'], train_df['loss'], 'b-', label='Train')
    if val_csv and os.path.exists(val_csv):
        val_df = pd.read_csv(val_csv)
        axes[0, 0].plot(val_df['epoch'], val_df['loss'], 'r--', label='Val')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Average Dice
    axes[0, 1].plot(train_df['epoch'], train_df['avg_dice'], 'b-', label='Train')
    if val_csv and os.path.exists(val_csv):
        axes[0, 1].plot(val_df['epoch'], val_df['avg_dice'], 'r--', label='Val')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Dice Score')
    axes[0, 1].set_title('Average Dice Score')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    axes[0, 1].set_ylim(0, 1)
    
    # Average Precision/Recall
    axes[1, 0].plot(train_df['epoch'], train_df['avg_precision'], 'b-', label='Train Prec')
    axes[1, 0].plot(train_df['epoch'], train_df['avg_recall'], 'g-', label='Train Recall')
    if val_csv and os.path.exists(val_csv):
        axes[1, 0].plot(val_df['epoch'], val_df['avg_precision'], 'b--', label='Val Prec')
        axes[1, 0].plot(val_df['epoch'], val_df['avg_recall'], 'g--', label='Val Recall')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_title('Precision & Recall')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    axes[1, 0].set_ylim(0, 1)
    
    # Training time per epoch
    axes[1, 1].plot(train_df['epoch'], train_df['epoch_time_sec'] / 60, 'b-')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Time (minutes)')
    axes[1, 1].set_title('Epoch Duration')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    logging.getLogger().info(f"Training curves saved: {plot_path}")

# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def train_model(args):
    """Main training function with Accelerate"""
    # Initialize Accelerator
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )
    
    # Set seed for reproducibility
    set_seed(args.seed)
    set_determinism(args.seed)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f'Outputs/SegResNet_brats_accelerate_{timestamp}'
    
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        save_config(args, output_dir)
    
    logger = setup_logging(output_dir, accelerator)
    
    if accelerator.is_main_process:
        logger.info(f"="*80)
        logger.info(f"BraTS2020 SegResNet Training with Accelerate")
        logger.info(f"Number of processes: {accelerator.num_processes}")
        logger.info(f"Mixed Precision: {args.mixed_precision}")
        logger.info(f"="*80)
    
    # Data
    train_loader, val_loader, test_loader = create_data_loaders(args, accelerator, output_dir)
    
    # Model
    model = create_model(args)
    
    # Training components
    criterion = DiceFocalLoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=True)
    if args.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Prepare everything with Accelerator
    model, optimizer, train_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, lr_scheduler
    )
    
    if val_loader:
        val_loader = accelerator.prepare(val_loader)
    if test_loader:
        test_loader = accelerator.prepare(test_loader)
    
    if accelerator.is_main_process:
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model: {total_params:,} parameters")
    
    # Initialize CSV loggers (only main process)
    if accelerator.is_main_process:
        train_csv = init_csv_logger(output_dir, 'train', args.class_names)
        val_csv = init_csv_logger(output_dir, 'val', args.class_names) if val_loader else None
        logger.info("Starting training loop...")
    
    val_metrics = None
    training_start = time.time()
    
    # Training loop
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        # Train
        train_loss, train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, accelerator, args
        )
        lr_scheduler.step()
        
        epoch_time = time.time() - epoch_start
        
        if accelerator.is_main_process:
            print_metrics_table(train_metrics, args.class_names, epoch, args.epochs, 'TRAIN')
            print(f"Train Loss: {train_loss:.4f} | Time: {epoch_time:.2f}s")
            log_to_csv(train_csv, epoch, train_loss, train_metrics, epoch_time, args.class_names)
        
        # Validate
        if val_loader and epoch % args.val_interval == 0:
            val_loss, val_metrics = eval_epoch(model, val_loader, criterion, accelerator, args)
            
            if accelerator.is_main_process:
                print_metrics_table(val_metrics, args.class_names, epoch, args.epochs, 'VAL')
                print(f"Val Loss: {val_loss:.4f}")
                log_to_csv(val_csv, epoch, val_loss, val_metrics, 0, args.class_names)
        
        # Save checkpoint
        if accelerator.is_main_process and epoch % args.save_checkpoint_every == 0:
            save_checkpoint(model, optimizer, epoch, output_dir, args, accelerator)
    
    # Final test evaluation
    if accelerator.is_main_process:
        logger.info("\n" + "="*80)
        logger.info("Running final TEST evaluation...")
        logger.info("="*80)
    
    if test_loader:
        test_csv = init_csv_logger(output_dir, 'test', args.class_names) if accelerator.is_main_process else None
        test_loss, test_metrics = eval_epoch(model, test_loader, criterion, accelerator, args)
        
        if accelerator.is_main_process:
            print_metrics_table(test_metrics, args.class_names, args.epochs, args.epochs, 'TEST')
            print(f"Test Loss: {test_loss:.4f}")
            log_to_csv(test_csv, args.epochs, test_loss, test_metrics, 0, args.class_names)
    
    # Finalize
    if accelerator.is_main_process:
        total_time = time.time() - training_start
        logger.info(f"\nTotal training time: {total_time:.2f}s ({total_time/3600:.2f}h)")
        
        # Save final model
        save_checkpoint(model, optimizer, args.epochs, output_dir, args, accelerator, is_final=True)
        
        # Save summary
        summary = {
            'total_epochs': args.epochs,
            'total_time_seconds': total_time,
            'total_time_hours': total_time / 3600,
            'batch_size_per_gpu': args.batch_size,
            'num_processes': accelerator.num_processes,
            'mixed_precision': args.mixed_precision,
            'final_train_loss': train_loss,
            'final_train_metrics': train_metrics,
        }
        if val_loader and val_metrics is not None:
            summary['final_val_metrics'] = val_metrics
        if test_loader:
            summary['final_test_loss'] = test_loss
            summary['final_test_metrics'] = test_metrics
        
        with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=4)
        
        # Plot curves
        if args.plot_metrics:
            plot_training_curves(train_csv, val_csv, output_dir)
        
        logger.info(f"\nTraining complete! Results in: {output_dir}")
    
    return output_dir

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')
    os.environ['MONAI_HIDE_WELCOME'] = '1'
    
    parser = argparse.ArgumentParser(description="BraTS 3D Segmentation with Accelerate")
    
    # Data
    parser.add_argument('--data_dir', type=str, 
                        default='Data/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData')
    parser.add_argument('--roi_size', type=int, nargs=3, default=[128, 128, 128],
                        help='ROI size for inference')
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--train_split', type=float, default=0.80, help='Train split (80%)')
    parser.add_argument('--val_split', type=float, default=0.05, help='Val split (5%)')
    parser.add_argument('--test_split', type=float, default=0.15, help='Test split (15%)')
    
    # Model
    parser.add_argument('--in_channels', type=int, default=4)
    parser.add_argument('--init_filters', type=int, default=16, help='Initial filters')
    
    # Training
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['Adam', 'SGD'])
    parser.add_argument('--batch_size', type=int, default=2, help='Per GPU')
    parser.add_argument('--val_batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--val_interval', type=int, default=5, help='Validate every N epochs')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--mixed_precision', type=str, default='fp16', 
                        choices=['no', 'fp16', 'bf16'], help='Mixed precision training')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='Gradient accumulation steps')
    
    # Hardware
    parser.add_argument('--num_workers', type=int, default=8, help='Data loader workers')
    parser.add_argument('--pin_memory', action='store_true', default=True)
    parser.add_argument('--persistent_workers', action='store_true', default=True)
    parser.add_argument('--prefetch_factor', type=int, default=4, help='Prefetch batches')
    parser.add_argument('--sw_batch_size', type=int, default=2)
    parser.add_argument('--use_cache', action='store_true', default=False,
                        help='Cache dataset in RAM (faster but uses memory)')
    
    # Output
    parser.add_argument('--save_checkpoint_every', type=int, default=10)
    parser.add_argument('--plot_metrics', action='store_true', default=True)
    
    args = parser.parse_args()
    
    args.roi_size = tuple(args.roi_size)
    args.class_names = ['Tumor Core (TC)', 'Whole Tumor (WT)', 'Enhancing Tumor (ET)']
    
    try:
        output_dir = train_model(args)
        print(f"\n✓ Training completed successfully!")
        print(f"✓ Results: {output_dir}")
    except Exception as e:
        print(f"\n✗ Training failed: {str(e)}")
        raise e
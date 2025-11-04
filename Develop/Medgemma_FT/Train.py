import os
import csv
import json
import time
import random
import argparse
from pathlib import Path
from datetime import datetime

# Import unsloth first for optimizations
from unsloth import FastVisionModel
from unsloth import get_chat_template
from unsloth.trainer import UnslothVisionDataCollator

# Other imports
import torch
from tqdm import tqdm
from PIL import Image
from datasets import load_dataset
from rouge_score import rouge_scorer
from trl import SFTTrainer, SFTConfig
from sklearn.model_selection import train_test_split

# ============================================================================
# PYTORCH CONFIGURATION
# ============================================================================
os.environ['TORCHINDUCTOR_CACHE_DIR'] = os.path.expanduser('~/.torchinductor_cache')
os.environ['TORCH_COMPILE_DISABLE'] = '1'
os.environ['TORCHDYNAMO_DISABLE'] = '1'

# ============================================================================
# ARGUMENT PARSER
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description='Fine-tune MedGemma on BraTS dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=1, 
                       help='Number of training epochs')
    parser.add_argument('--steps', type=int, default=None, 
                       help='Max training steps (overrides epochs if set)')
    parser.add_argument('--train_split', type=float, default=0.95, 
                       help='Train/test split ratio (0.95 = 95%% train, 5%% test)')
    parser.add_argument('--val_samples', type=int, default=3,
                       help='Number of random test samples to use for validation')
    parser.add_argument('--val_every_steps', type=int, default=10,
                       help='Run validation every N steps during training')
    parser.add_argument('--skip_full_test', action='store_true',
                       help='Skip full test set evaluation (only run validation samples)')
    
    # Model saving
    parser.add_argument('--model_name', type=str, default='brats_medgemma',
                       help='Model name for saving locally')
    
    # Paths
    parser.add_argument('--data_dir', type=str, default='Data/Brats_slices',
                       help='Data directory containing patient folders')
    parser.add_argument('--output_dir', type=str, default='outputs_medgemma_brats',
                       help='Output directory for models and logs')
    
    # LoRA parameters
    parser.add_argument('--lora_r', type=int, default=1,
                       help='LoRA rank (higher = more parameters)')
    parser.add_argument('--lora_alpha', type=int, default=32,
                       help='LoRA alpha scaling')
    
    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=2,
                       help='Per-device batch size')
    parser.add_argument('--accum_steps', type=int, default=4,
                       help='Gradient accumulation steps')
    parser.add_argument('--lr', type=float, default=2e-4,
                       help='Learning rate')
    parser.add_argument('--warmup_ratio', type=float, default=0.03,
                       help='Warmup ratio')
    
    # Generation parameters
    parser.add_argument('--max_new_tokens', type=int, default=128,
                       help='Max tokens to generate during inference')
    parser.add_argument('--temperature', type=float, default=0.7,
                       help='Sampling temperature')
    
    # Evaluation parameters
    parser.add_argument('--eval_batch_size', type=int, default=4,
                       help='Batch size for test evaluation (higher = faster but more memory)')
    
    return parser.parse_args()

# ============================================================================
# GLOBAL CONFIG
# ============================================================================
args = parse_args()

MODEL_NAME = "unsloth/medgemma-4b-it"
DATA_DIR = args.data_dir

# Timestamp for this run
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# Create output directory with model name and timestamp
OUTPUT_DIR = os.path.join(args.output_dir, f"{args.model_name}_{TIMESTAMP}")

LORA_DIR = os.path.join(OUTPUT_DIR, "lora_adapters")
MERGED_DIR = os.path.join(OUTPUT_DIR, "merged_model")
METRICS_DIR = os.path.join(OUTPUT_DIR, "metrics")

# Create directories
for dir_path in [OUTPUT_DIR, LORA_DIR, MERGED_DIR, METRICS_DIR]:
    Path(dir_path).mkdir(parents=True, exist_ok=True)

SEED = 3407
random.seed(SEED)

# ============================================================================
# PROMPT & ANSWER TEMPLATES FOR DATA AUGMENTATION
# ============================================================================
SEGMENTATION_LEGEND = """
Segmentation colors indicate:
- RED: Non-Enhancing Tumor Core (NCR/NET) - necrotic tumor tissue
- GREEN: Peritumoral Edema (ED) - swelling around the tumor
- BLUE: Enhancing Tumor (ET) - actively growing tumor tissue"""

PROMPT_TEMPLATES = {
    "single_slice": [
        f"Describe this brain MRI slice in detail. {SEGMENTATION_LEGEND} Focus on tumor regions and their characteristics.",
        f"Analyze this axial T1CE brain scan. {SEGMENTATION_LEGEND} Identify all tumor components present.",
        f"Provide a detailed medical description of this brain scan. {SEGMENTATION_LEGEND} Note lesions and abnormalities.",
        f"Examine this MRI slice and describe tumor characteristics. {SEGMENTATION_LEGEND} Include necrosis and edema assessment.",
    ],
    "three_slices": [
        f"Describe tumor progression across these 3 consecutive brain MRI slices (10-slice intervals). {SEGMENTATION_LEGEND}",
        f"Analyze spatial distribution of tumor components in these 3 axial slices with 10-slice spacing. {SEGMENTATION_LEGEND}",
        f"Examine these 3 MRI slices and describe how the lesion evolves through brain volume. {SEGMENTATION_LEGEND}",
        f"Provide comprehensive analysis of tumor across these 3 sequential slices at 10-slice intervals. {SEGMENTATION_LEGEND}",
    ],
    "five_slices": [
        f"Analyze these 5 consecutive MRI slices (5-slice intervals) and describe complete tumor profile. {SEGMENTATION_LEGEND}",
        f"Describe tumor characteristics across these 5 sequential brain scans with 5-slice spacing. {SEGMENTATION_LEGEND}",
        f"Provide detailed analysis of lesion distribution visible in these 5 slices. {SEGMENTATION_LEGEND}",
        f"Examine these 5 axial slices and characterize tumor extent, necrosis, edema, and enhancing regions. {SEGMENTATION_LEGEND}",
    ]
}

# Answer prefixes to create synthetic data variation
ANSWER_TEMPLATES = {
    "single_slice": [
        "This single axial T1CE MRI slice shows: ",
        "Analysis of this brain scan reveals: ",
        "The MRI slice demonstrates: ",
        "Examining this single slice, I observe: ",
    ],
    "three_slices": [
        "Across these 3 consecutive slices (10-slice intervals), the tumor shows: ",
        "Analysis of these 3 sequential brain scans reveals: ",
        "The tumor progression through these 3 slices demonstrates: ",
        "Examining these 3 axial slices with 10-slice spacing: ",
    ],
    "five_slices": [
        "Across these 5 consecutive slices (5-slice intervals), the complete tumor profile shows: ",
        "Analysis of these 5 sequential MRI scans reveals: ",
        "The comprehensive tumor distribution through these 5 slices demonstrates: ",
        "Examining these 5 axial slices with 5-slice spacing: ",
    ]
}

def get_instruction_for_image_type(image_type, index):
    """Get systematic instruction prompt based on image type and index."""
    if image_type in PROMPT_TEMPLATES:
        prompts = PROMPT_TEMPLATES[image_type]
        return prompts[index % len(prompts)]
    return f"Describe the medical image in detail. {SEGMENTATION_LEGEND}"

def get_answer_prefix_for_image_type(image_type, index):
    """Get systematic answer prefix based on image type and index."""
    if image_type in ANSWER_TEMPLATES:
        prefixes = ANSWER_TEMPLATES[image_type]
        return prefixes[index % len(prefixes)]
    return ""

# ============================================================================
# DATA PREPARATION
# ============================================================================
def load_image_for_sample(patient_folder, image_type):
    """Load a specific image type for a patient."""
    image_name = f"{image_type}.jpg"
    image_path = os.path.join(DATA_DIR, patient_folder, image_name)
    if os.path.exists(image_path):
        try:
            return Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Warning: Could not load {image_path}: {e}")
    return None

def convert_to_conversation(sample, patient_folder, image_type, sample_index):
    """Convert a sample to conversation format with systematic prompt/answer variation."""
    image = load_image_for_sample(patient_folder, image_type)
    if image is None:
        return None
    
    # Get systematic prompt and answer prefix based on index
    instruction = get_instruction_for_image_type(image_type, sample_index)
    answer_prefix = get_answer_prefix_for_image_type(image_type, sample_index)
    
    # Combine answer prefix with original text
    enhanced_answer = answer_prefix + sample["text"]
    
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": instruction},
                {"type": "image", "image": image},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": enhanced_answer}]},
    ]
    
    # Return with metadata for tracking
    image_path = os.path.join(DATA_DIR, patient_folder, f"{image_type}.jpg")
    return {
        "messages": conversation,
        "patient_id": patient_folder,
        "image_type": image_type,
        "metadata": {
            "original_textbrats_id": sample_index,
            "image_path": image_path,
            "prompt_template_index": sample_index % len(PROMPT_TEMPLATES[image_type]),
            "answer_template_index": sample_index % len(ANSWER_TEMPLATES[image_type]),
            "full_prompt": instruction,
            "answer_prefix": answer_prefix,
            "original_text": sample["text"],
            "enhanced_answer": enhanced_answer
        }
    }

def prepare_dataset(train_split=0.95):
    """Prepare train and test datasets with comprehensive tracking."""
    print("\n" + "="*80)
    print("📊 PREPARING DATASET")
    print("="*80)
    
    # Load TextBraTS dataset
    dataset = load_dataset("Jupitern52/TextBraTS")
    print(f"✓ Loaded TextBraTS dataset: {len(dataset['train'])} samples")
    
    # Get patient folders
    patient_folders = sorted([f for f in os.listdir(DATA_DIR) if f.startswith('BraTS20_Training_')])
    print(f"✓ Found {len(patient_folders)} patient folders")
    
    # Convert dataset with 3x augmentation (single, three, five slices)
    # Each image type gets systematic prompt/answer variations
    converted_dataset = []
    image_types = ["single_slice", "three_slices", "five_slices"]
    
    print("✓ Converting samples (3x augmentation per patient with systematic variations)...")
    for i, sample in enumerate(tqdm(dataset['train'], desc="Processing")):
        if i >= len(patient_folders):
            break
        patient_folder = patient_folders[i]
        
        for img_type in image_types:
            conv = convert_to_conversation(sample, patient_folder, img_type, i)
            if conv:
                converted_dataset.append(conv)
    
    print(f"✓ Total converted samples: {len(converted_dataset)}")
    
    # Split into train and test
    train_ds, test_ds = train_test_split(
        converted_dataset, 
        test_size=(1-train_split), 
        random_state=SEED
    )
    
    # Build comprehensive data distribution documentation
    data_distribution = build_data_distribution_doc(
        dataset, patient_folders, converted_dataset, train_ds, test_ds, train_split
    )
    
    print(f"\n📈 Split Summary:")
    print(f"  Train: {len(train_ds)} samples ({train_split*100:.1f}%)")
    print(f"  Test:  {len(test_ds)} samples ({(1-train_split)*100:.1f}%)")
    print("="*80 + "\n")
    
    return train_ds, test_ds, data_distribution

def build_data_distribution_doc(dataset, patient_folders, converted_dataset, train_ds, test_ds, train_split):
    """Build comprehensive data distribution documentation."""
    
    # Collect all samples with full details
    all_samples = []
    for idx, sample in enumerate(converted_dataset):
        meta = sample.get('metadata', {})
        split = 'train' if sample in train_ds else 'test'
        
        all_samples.append({
            "sample_id": idx,
            "split": split,
            "patient_id": sample['patient_id'],
            "image_type": sample['image_type'],
            "image_path": meta.get('image_path', ''),
            "original_textbrats_id": meta.get('original_textbrats_id', -1),
            "prompt_template_index": meta.get('prompt_template_index', -1),
            "answer_template_index": meta.get('answer_template_index', -1),
            "full_prompt_text": meta.get('full_prompt', ''),
            "answer_prefix": meta.get('answer_prefix', ''),
            "original_text": meta.get('original_text', ''),
            "enhanced_answer": meta.get('enhanced_answer', '')
        })
    
    # Calculate statistics
    train_samples = [s for s in all_samples if s['split'] == 'train']
    test_samples = [s for s in all_samples if s['split'] == 'test']
    
    # Patient distribution
    train_patients = {}
    test_patients = {}
    for s in train_samples:
        train_patients[s['patient_id']] = train_patients.get(s['patient_id'], 0) + 1
    for s in test_samples:
        test_patients[s['patient_id']] = test_patients.get(s['patient_id'], 0) + 1
    
    # Image type distribution
    train_img_types = {}
    test_img_types = {}
    for s in train_samples:
        train_img_types[s['image_type']] = train_img_types.get(s['image_type'], 0) + 1
    for s in test_samples:
        test_img_types[s['image_type']] = test_img_types.get(s['image_type'], 0) + 1
    
    # Build comprehensive documentation
    distribution = {
        "experiment_metadata": {
            "timestamp": TIMESTAMP,
            "random_seed": SEED,
            "source_dataset": "Jupitern52/TextBraTS",
            "data_directory": DATA_DIR
        },
        
        "configuration": {
            "train_split_ratio": train_split,
            "augmentation_strategy": "3x per patient (single_slice + three_slices + five_slices)",
            "image_types": ["single_slice", "three_slices", "five_slices"],
            "prompt_templates_per_type": {k: len(v) for k, v in PROMPT_TEMPLATES.items()},
            "answer_templates_per_type": {k: len(v) for k, v in ANSWER_TEMPLATES.items()}
        },
        
        "prompt_templates": {
            "segmentation_legend": SEGMENTATION_LEGEND,
            "templates": {
                img_type: {
                    "prompts": PROMPT_TEMPLATES[img_type],
                    "answer_prefixes": ANSWER_TEMPLATES[img_type]
                }
                for img_type in ["single_slice", "three_slices", "five_slices"]
            }
        },
        
        "data_summary": {
            "original_textbrats_samples": len(dataset['train']),
            "available_patients": len(patient_folders),
            "patients_used": len([p for p in patient_folders if any(s['patient_id'] == p for s in all_samples)]),
            "total_created_samples": len(all_samples),
            "train_samples": len(train_samples),
            "test_samples": len(test_samples)
        },
        
        "train_split": {
            "total_samples": len(train_samples),
            "percentage": train_split * 100,
            "unique_patients": len(train_patients),
            "patient_distribution": train_patients,
            "image_type_distribution": train_img_types,
            "samples_per_image_type": {
                "single_slice": train_img_types.get('single_slice', 0),
                "three_slices": train_img_types.get('three_slices', 0),
                "five_slices": train_img_types.get('five_slices', 0)
            }
        },
        
        "test_split": {
            "total_samples": len(test_samples),
            "percentage": (1 - train_split) * 100,
            "unique_patients": len(test_patients),
            "patient_distribution": test_patients,
            "image_type_distribution": test_img_types,
            "samples_per_image_type": {
                "single_slice": test_img_types.get('single_slice', 0),
                "three_slices": test_img_types.get('three_slices', 0),
                "five_slices": test_img_types.get('five_slices', 0)
            }
        },
        
        "complete_sample_list": all_samples
    }
    
    return distribution

# ============================================================================
# MODEL SETUP
# ============================================================================
def setup_model(lora_r=4, lora_alpha=4):
    """Setup model with LoRA adapters."""
    print("\n" + "="*80)
    print("🤖 SETTING UP MODEL")
    print("="*80)
    
    # Load base model
    print(f"Loading: {MODEL_NAME}")
    model, processor = FastVisionModel.from_pretrained(
        MODEL_NAME,
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
    )
    print(f"✓ Model loaded")
    
    # Apply chat template
    processor = get_chat_template(processor, "gemma-3")
    print("✓ Chat template applied")
    
    # Add LoRA adapters
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0,
        bias="none",
        random_state=SEED,
        use_rslora=False,
        loftq_config=None,
        target_modules="all-linear",
    )
    print(f"✓ LoRA adapters added (r={lora_r}, alpha={lora_alpha})")
    print("="*80 + "\n")
    
    return model, processor

# ============================================================================
# PERIODIC VALIDATION CALLBACK
# ============================================================================
from transformers import TrainerCallback

class PeriodicValidationCallback(TrainerCallback):
    """Callback to run validation during training at regular intervals."""
    
    def __init__(self, model, processor, test_dataset, val_samples, val_every_steps, output_dir):
        self.model = model
        self.processor = processor
        self.test_dataset = test_dataset
        self.val_samples = val_samples
        self.val_every_steps = val_every_steps
        self.output_dir = output_dir
        self.validation_log = os.path.join(output_dir, f"validation_log_{TIMESTAMP}.csv")
        self.validation_details = []
        
        # Initialize validation log CSV
        with open(self.validation_log, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['step', 'epoch', 'sample_id', 'patient_id', 'image_type', 
                           'rouge1', 'rouge2', 'rougeL', 'avg_rouge'])
    
    def on_step_end(self, args, state, control, **kwargs):
        """Run validation at specified intervals."""
        if state.global_step % self.val_every_steps == 0 and state.global_step > 0:
            self._run_validation(state.global_step, state.epoch)
    
    def _run_validation(self, step, epoch):
        """Run validation on random samples."""
        print(f"\n{'─'*80}")
        print(f"🔍 VALIDATION AT STEP {step} (Epoch {epoch:.2f})")
        print(f"{'─'*80}")
        
        # Set model to inference mode temporarily
        self.model.eval()
        FastVisionModel.for_inference(self.model)
        
        # Select random samples
        random_indices = random.sample(range(len(self.test_dataset)), 
                                      min(self.val_samples, len(self.test_dataset)))
        
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        step_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        
        for idx in random_indices:
            sample = self.test_dataset[idx]
            messages = sample["messages"]
            image = messages[0]["content"][1]["image"]
            instruction = messages[0]["content"][0]["text"]
            ground_truth = messages[1]["content"][0]["text"]
            
            # Prepare input
            user_messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {"type": "image", "image": image},
                ],
            }]
            input_text = self.processor.apply_chat_template(user_messages, add_generation_prompt=True)
            inputs = self.processor(image, input_text, add_special_tokens=False, return_tensors="pt").to("cuda")
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                    temperature=args.temperature,
                    top_p=0.95,
                    top_k=40,
                    do_sample=True,
                    repetition_penalty=1.1,
                )
            
            generated = self.processor.decode(outputs[0][inputs['input_ids'].shape[1]:], 
                                             skip_special_tokens=True).strip()
            
            # Calculate ROUGE scores
            scores = scorer.score(ground_truth, generated)
            
            for key in ['rouge1', 'rouge2', 'rougeL']:
                step_scores[key].append(scores[key].fmeasure)
            
            avg_rouge = sum([scores[k].fmeasure for k in scores]) / 3
            
            # Log to CSV
            with open(self.validation_log, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    step, f"{epoch:.4f}", idx, 
                    sample.get('patient_id', 'N/A'),
                    sample.get('image_type', 'N/A'),
                    f"{scores['rouge1'].fmeasure:.4f}",
                    f"{scores['rouge2'].fmeasure:.4f}",
                    f"{scores['rougeL'].fmeasure:.4f}",
                    f"{avg_rouge:.4f}"
                ])
            
            # Store full details
            self.validation_details.append({
                'step': step,
                'epoch': epoch,
                'sample_id': idx,
                'patient_id': sample.get('patient_id', 'N/A'),
                'image_type': sample.get('image_type', 'N/A'),
                'instruction': instruction,
                'ground_truth': ground_truth,
                'generated': generated,
                'rouge1': scores['rouge1'].fmeasure,
                'rouge2': scores['rouge2'].fmeasure,
                'rougeL': scores['rougeL'].fmeasure
            })
        
        # Print averages
        avg_scores = {key: sum(scores)/len(scores) for key, scores in step_scores.items()}
        print(f"\n📊 Validation Results:")
        for key, score in avg_scores.items():
            print(f"  {key.upper()}: {score:.4f}")
        print(f"{'─'*80}\n")
        
        # Return to training mode
        self.model.train()
        FastVisionModel.for_training(self.model)

class TrainingLogger(TrainerCallback):
    """Callback to log training metrics per step to CSV."""
    
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.log_file = os.path.join(output_dir, f"training_log_{TIMESTAMP}.csv")
        self.start_time = time.time()
        
        # Initialize CSV
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['step', 'epoch', 'loss', 'learning_rate', 'grad_norm', 'elapsed_time_s', 'timestamp'])
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when logging happens."""
        if logs is None:
            return
        
        # Only log training steps (not eval or other logs)
        if 'loss' in logs and state.global_step > 0:
            elapsed = time.time() - self.start_time
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            step = state.global_step
            loss = logs.get('loss', 0.0)
            lr = logs.get('learning_rate', 0.0)
            grad_norm = logs.get('grad_norm', 0.0)
            epoch = logs.get('epoch', 0.0)
            
            with open(self.log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([step, f"{epoch:.4f}", f"{loss:.6f}", f"{lr:.2e}", f"{grad_norm:.4f}", f"{elapsed:.2f}", timestamp])

# ============================================================================
# TRAINING WITH PERIODIC VALIDATION
# ============================================================================
def train_model(model, processor, train_dataset, test_dataset, num_epochs=1, max_steps=None):
    """Train the model with comprehensive logging and periodic validation."""
    print("\n" + "="*80)
    print("🚀 TRAINING MODEL")
    print("="*80)
    
    FastVisionModel.for_training(model)
    
    # Calculate training steps
    effective_batch = args.batch_size * args.accum_steps
    steps_per_epoch = len(train_dataset) // effective_batch
    
    if max_steps is None:
        total_steps = steps_per_epoch * num_epochs
    else:
        total_steps = max_steps
        num_epochs = max_steps / steps_per_epoch
    
    print(f"📋 Training Configuration:")
    print(f"  Dataset: {len(train_dataset)} samples")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Gradient accumulation: {args.accum_steps}")
    print(f"  Effective batch: {effective_batch}")
    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Epochs: {num_epochs if max_steps is None else f'{num_epochs:.2f}'}")
    print(f"  Total steps: {total_steps}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Warmup ratio: {args.warmup_ratio}")
    print(f"  Validation every: {args.val_every_steps} steps")
    print()
    
    # Initialize callbacks
    training_logger = TrainingLogger(METRICS_DIR)
    validation_callback = PeriodicValidationCallback(
        model, processor, test_dataset, 
        args.val_samples, args.val_every_steps, METRICS_DIR
    )
    
    # Setup trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        processing_class=processor.tokenizer,
        data_collator=UnslothVisionDataCollator(model, processor),
        callbacks=[training_logger, validation_callback],
        args=SFTConfig(
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.accum_steps,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            max_grad_norm=0.3,
            warmup_ratio=args.warmup_ratio,
            num_train_epochs=num_epochs,
            max_steps=total_steps,
            learning_rate=args.lr,
            logging_steps=1,
            save_strategy="steps",
            save_steps=total_steps,
            optim="adamw_torch_fused",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=SEED,
            output_dir=OUTPUT_DIR,
            report_to="none",
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            max_length=2048,
        ),
    )
    
    # GPU stats
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"🎮 GPU: {gpu_stats.name}")
    print(f"💾 Max memory: {max_memory} GB")
    print(f"📊 Reserved: {start_gpu_memory} GB\n")
    
    # Train
    print("⏳ Training started...")
    start_time = time.time()
    trainer_stats = trainer.train()
    training_time = time.time() - start_time
    
    # Final memory
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    
    print("\n" + "-"*80)
    print("✅ TRAINING COMPLETED")
    print("-"*80)
    print(f"⏱️  Time: {training_time:.2f}s ({training_time/60:.2f} min)")
    print(f"💾 Peak memory: {used_memory} GB")
    print(f"📈 LoRA memory: {used_memory_for_lora} GB")
    print(f"📉 Final loss: {trainer_stats.metrics.get('train_loss', 'N/A')}")
    print(f"📝 Training log: {training_logger.log_file}")
    print(f"📝 Validation log: {validation_callback.validation_log}")
    print("="*80 + "\n")
    
    return trainer, trainer_stats, training_time, validation_callback.validation_details

# ============================================================================
# FULL TEST EVALUATION
# ============================================================================
def evaluate_test_set(model, processor, test_dataset, batch_size=4):
    """Evaluate on full test set with ROUGE scores using batch processing."""
    print("\n" + "="*80)
    print("📊 FULL TEST SET EVALUATION (BATCH PROCESSING)")
    print("="*80)
    
    FastVisionModel.for_inference(model)
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    all_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
    test_details = []
    
    print(f"Evaluating {len(test_dataset)} test samples...")
    print(f"Batch size: {batch_size}")
    print()
    
    start_time = time.time()
    
    # Process in batches
    num_batches = (len(test_dataset) + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(num_batches), desc="Testing Batches"):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(test_dataset))
        batch_samples = test_dataset[batch_start:batch_end]
        
        # Prepare batch data
        batch_images = []
        batch_instructions = []
        batch_ground_truths = []
        batch_metadata = []
        
        for i, sample in enumerate(batch_samples):
            messages = sample["messages"]
            image = messages[0]["content"][1]["image"]
            instruction = messages[0]["content"][0]["text"]
            ground_truth = messages[1]["content"][0]["text"]
            
            batch_images.append(image)
            batch_instructions.append(instruction)
            batch_ground_truths.append(ground_truth)
            batch_metadata.append({
                'sample_id': batch_start + i,
                'patient_id': sample.get('patient_id', 'N/A'),
                'image_type': sample.get('image_type', 'N/A')
            })
        
        # Process batch - prepare inputs for each sample
        batch_input_ids = []
        batch_attention_masks = []
        batch_pixel_values = []
        
        for image, instruction in zip(batch_images, batch_instructions):
            user_messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {"type": "image", "image": image},
                ],
            }]
            input_text = processor.apply_chat_template(user_messages, add_generation_prompt=True)
            inputs = processor(image, input_text, add_special_tokens=False, return_tensors="pt")
            
            batch_input_ids.append(inputs['input_ids'])
            batch_attention_masks.append(inputs['attention_mask'])
            batch_pixel_values.append(inputs['pixel_values'])
        
        # Pad sequences to same length within batch
        max_length = max(ids.shape[1] for ids in batch_input_ids)
        padded_input_ids = []
        padded_attention_masks = []
        original_lengths = []
        
        for input_ids, attention_mask in zip(batch_input_ids, batch_attention_masks):
            original_lengths.append(input_ids.shape[1])
            pad_length = max_length - input_ids.shape[1]
            
            if pad_length > 0:
                padded_input_ids.append(torch.cat([
                    input_ids,
                    torch.zeros((1, pad_length), dtype=input_ids.dtype)
                ], dim=1))
                padded_attention_masks.append(torch.cat([
                    attention_mask,
                    torch.zeros((1, pad_length), dtype=attention_mask.dtype)
                ], dim=1))
            else:
                padded_input_ids.append(input_ids)
                padded_attention_masks.append(attention_mask)
        
        # Stack into batch tensors
        batch_inputs = {
            'input_ids': torch.cat(padded_input_ids, dim=0).to("cuda"),
            'attention_mask': torch.cat(padded_attention_masks, dim=0).to("cuda"),
            'pixel_values': torch.cat(batch_pixel_values, dim=0).to("cuda")
        }
        
        # Generate for entire batch
        with torch.no_grad():
            batch_outputs = model.generate(
                **batch_inputs,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                temperature=args.temperature,
                top_p=0.95,
                top_k=40,
                do_sample=True,
                repetition_penalty=1.1,
            )
        
        # Decode each output in the batch
        for i, (output, orig_len, ground_truth, metadata) in enumerate(
            zip(batch_outputs, original_lengths, batch_ground_truths, batch_metadata)
        ):
            generated = processor.decode(output[orig_len:], skip_special_tokens=True).strip()
            
            # Calculate ROUGE scores
            scores = scorer.score(ground_truth, generated)
            
            for key in ['rouge1', 'rouge2', 'rougeL']:
                all_scores[key].append(scores[key].fmeasure)
            
            # Store details
            test_details.append({
                'sample_id': metadata['sample_id'],
                'patient_id': metadata['patient_id'],
                'image_type': metadata['image_type'],
                'rouge1': scores['rouge1'].fmeasure,
                'rouge2': scores['rouge2'].fmeasure,
                'rougeL': scores['rougeL'].fmeasure,
                'ground_truth': ground_truth,
                'generated': generated
            })
    
    eval_time = time.time() - start_time
    
    # Calculate averages
    avg_scores = {key: sum(scores)/len(scores) for key, scores in all_scores.items()}
    
    print("\n" + "─"*80)
    print("📈 TEST RESULTS")
    print("─"*80)
    for key, score in avg_scores.items():
        metric_name = key.upper() if key != 'bleu' else 'BLEU'
        print(f"  {metric_name}: {score:.4f}")
    print(f"\n⏱️  Evaluation time: {eval_time:.2f}s ({eval_time/60:.2f} min)")
    print("="*80 + "\n")
    
    return avg_scores, test_details

# ============================================================================
# SAVE METRICS TO CSV
# ============================================================================
def save_metrics_csv(train_stats, test_scores, test_details, training_time, validation_details, train_size, test_size, data_distribution):
    """Save all metrics to CSV files and comprehensive data distribution JSON."""
    print("\n" + "="*80)
    print("💾 SAVING METRICS & DATA DISTRIBUTION")
    print("="*80)
    
    # 1. Training summary
    train_summary = os.path.join(METRICS_DIR, f"training_summary_{TIMESTAMP}.csv")
    with open(train_summary, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        writer.writerow(['Timestamp', TIMESTAMP])
        writer.writerow(['Training Time (s)', f"{training_time:.2f}"])
        writer.writerow(['Training Time (min)', f"{training_time/60:.2f}"])
        writer.writerow(['Final Loss', train_stats.metrics.get('train_loss', 'N/A')])
        writer.writerow(['Epochs', args.epochs])
        writer.writerow(['Steps', args.steps if args.steps else 'Full epochs'])
        writer.writerow(['Train Samples', train_size])
        writer.writerow(['Test Samples', test_size])
        writer.writerow(['Train Split', args.train_split])
        writer.writerow(['Learning Rate', args.lr])
        writer.writerow(['LoRA Rank', args.lora_r])
        writer.writerow(['LoRA Alpha', args.lora_alpha])
        writer.writerow(['Batch Size', args.batch_size])
        writer.writerow(['Gradient Accumulation', args.accum_steps])
        writer.writerow(['Validation Every N Steps', args.val_every_steps])
    print(f"✓ Training summary: {train_summary}")
    
    # 2. Test summary
    if test_scores is not None:
        test_summary = os.path.join(METRICS_DIR, f"test_summary_{TIMESTAMP}.csv")
        with open(test_summary, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Score'])
            for key, value in test_scores.items():
                writer.writerow([key.upper(), f"{value:.6f}"])
        print(f"✓ Test summary: {test_summary}")
    else:
        print("⏩ Test summary: Skipped (--skip_full_test enabled)")
    
    # 3. Detailed test results
    if test_details:
        test_detailed = os.path.join(METRICS_DIR, f"test_detailed_{TIMESTAMP}.csv")
        with open(test_detailed, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['sample_id', 'patient_id', 'image_type', 
                                                    'rouge1', 'rouge2', 'rougeL',
                                                    'ground_truth', 'generated'])
            writer.writeheader()
            writer.writerows(test_details)
        print(f"✓ Test detailed: {test_detailed}")
    else:
        print("⏩ Test detailed: Skipped (--skip_full_test enabled)")
    
    # 4. Validation samples during training (JSON for readability)
    if validation_details:
        val_json = os.path.join(METRICS_DIR, f"validation_during_training_{TIMESTAMP}.json")
        with open(val_json, 'w') as f:
            json.dump(validation_details, f, indent=2)
        print(f"✓ Validation during training: {val_json}")
    
    # 5. SIMPLIFIED DATA DISTRIBUTION - ONLY WHAT'S NEEDED FOR PLOTTING
    if data_distribution:
        # Create simplified version for plotting - only essential data
        simplified_distribution = {
            "experiment_metadata": {
                "timestamp": data_distribution["experiment_metadata"]["timestamp"],
                "random_seed": data_distribution["experiment_metadata"]["random_seed"],
                "source_dataset": data_distribution["experiment_metadata"]["source_dataset"]
            },
            "data_summary": data_distribution["data_summary"],
            "train_split": {
                "image_type_distribution": data_distribution["train_split"]["image_type_distribution"]
            },
            "test_split": {
                "image_type_distribution": data_distribution["test_split"]["image_type_distribution"]
            }
        }
        
        # Save the simplified distribution for plotting
        complete_doc_path = os.path.join(METRICS_DIR, f"complete_data_distribution_{TIMESTAMP}.json")
        with open(complete_doc_path, 'w') as f:
            json.dump(simplified_distribution, f, indent=2)
        
        print(f"\n✅ SIMPLIFIED DATA DISTRIBUTION SAVED:")
        print(f"   📄 {complete_doc_path}")
        print(f"\n   Contains only plotting essentials:")
        print(f"   • Train/test sample counts")
        print(f"   • Image type distributions for visualization")
        print(f"   • Basic experiment metadata")
        print(f"   • No redundant sample details (saved separately)")
    
    print("="*80 + "\n")

# ============================================================================
# SAVE MODELS WITH ENHANCED MODEL CARD
# ============================================================================
def save_models(model, processor, train_size, test_size, test_scores):
    """Save LoRA and merged models locally and optionally to HuggingFace."""
    print("\n" + "="*80)
    print("💾 SAVING MODELS")
    print("="*80)
    
    # 1. Save LoRA adapters locally
    print("\n📦 Saving LoRA adapters...")
    model.save_pretrained(LORA_DIR)
    processor.save_pretrained(LORA_DIR)
    print(f"✓ LoRA saved locally: {LORA_DIR}")
    
    # 2. Save merged model locally (16-bit)
    print("\n🔗 Saving merged model (16-bit)...")
    model.save_pretrained_merged(MERGED_DIR, processor, save_method="merged_16bit")
    
    # Clean up adapter configuration files to make it a true base model
    adapter_files = [
        "adapter_config.json",
        "adapter_model.safetensors", 
        "adapter_model.bin"
    ]
    for adapter_file in adapter_files:
        adapter_path = os.path.join(MERGED_DIR, adapter_file)
        if os.path.exists(adapter_path):
            os.remove(adapter_path)
            print(f"✓ Removed adapter file: {adapter_file}")
    
    print(f"✓ Merged model saved locally: {MERGED_DIR}")
    print("\n" + "="*80 + "\n")

# ============================================================================
# MAIN PIPELINE
# ============================================================================
def main():
    """Main training pipeline."""
    
    # Print header
    print("\n" + "="*80)
    print("🧠 MEDGEMMA BRATS FINE-TUNING PIPELINE (ENHANCED)")
    print("="*80)
    print(f"🕐 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📅 Run ID: {TIMESTAMP}")
    print("\n📋 Configuration:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Steps: {args.steps if args.steps else 'Full epochs'}")
    print(f"  Train/Test split: {args.train_split:.0%}/{(1-args.train_split):.0%}")
    print(f"  Validation samples: {args.val_samples}")
    print(f"  Validation frequency: Every {args.val_every_steps} steps")
    print(f"  LoRA r={args.lora_r}, α={args.lora_alpha}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Batch size: {args.batch_size} (effective: {args.batch_size * args.accum_steps})")
    print(f"  Output: {OUTPUT_DIR}")
    print("="*80)
    
    # Step 1: Prepare dataset
    train_dataset, test_dataset, data_distribution = prepare_dataset(args.train_split)
    
    # Step 2: Setup model
    model, processor = setup_model(args.lora_r, args.lora_alpha)
    
    # Step 3: Train with periodic validation
    trainer, train_stats, training_time, validation_details = train_model(
        model, processor, train_dataset, test_dataset,
        args.epochs, args.steps
    )
    
    # Step 4: Full test evaluation (optional)
    if args.skip_full_test:
        print("\n" + "="*80)
        print("⏩ SKIPPING FULL TEST SET EVALUATION")
        print("="*80)
        print("✓ Only periodic validation samples were evaluated during training")
        print("="*80 + "\n")
        test_scores = None
        test_details = []
    else:
        test_scores, test_details = evaluate_test_set(model, processor, test_dataset, 
                                                       batch_size=args.eval_batch_size)
    
    # Step 5: Save metrics
    save_metrics_csv(train_stats, test_scores, test_details, training_time, 
                     validation_details, len(train_dataset), len(test_dataset), data_distribution)
    
    # Step 6: Save models
    save_models(model, processor, len(train_dataset), len(test_dataset), test_scores)
    
    # Final summary
    print("\n" + "="*80)
    print("✅ PIPELINE COMPLETED SUCCESSFULLY")
    print("="*80)
    print(f"🕐 Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📁 Output directory: {OUTPUT_DIR}")
    print(f"  📦 LoRA: {LORA_DIR}")
    print(f"  🔗 Merged: {MERGED_DIR}")
    print(f"  📊 Metrics: {METRICS_DIR}")
    print(f"\n📊 Training Summary:")
    print(f"  Total samples: {len(train_dataset) + len(test_dataset)}")
    print(f"  Train: {len(train_dataset)} ({args.train_split:.0%})")
    print(f"  Test: {len(test_dataset)} ({(1-args.train_split):.0%})")
    print(f"  Validation runs: ~{len(validation_details) // args.val_samples}")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
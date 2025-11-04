#!/usr/bin/env python3
"""
Simple model comparison test - Clean and focused.
Tests basic questions and images, saves to clean CSV.
"""

import os
import csv
import torch
from pathlib import Path
from datetime import datetime
from PIL import Image
from unsloth import FastVisionModel
from unsloth.chat_templates import get_chat_template

# Model configurations
MODELS_TO_TEST = [
    {
        "name": "original",
        "path": "unsloth/medgemma-4b-it",
        "is_base": True,
    },
    {
        "name": "r1_alpha2_epochs1",
        "path": "outputs_medgemma_brats/brats_medgemma_r1_alpha2_20251013_235648/merged_model",
        "is_base": False,
    },
    {
        "name": "r1_alpha4_epochs1",
        "path": "outputs_medgemma_brats/brats_medgemma_r1_alpha4_20251014_001744/merged_model",
        "is_base": False,
    },
    {
        "name": "r1_alpha4_epochs2",
        "path": "outputs_medgemma_brats/brats_medgemma_r1_alpha4_20251014_005739/merged_model",
        "is_base": False,
    },
    {
        "name": "r4_alpha4_epochs2",
        "path": "outputs_medgemma_brats/brats_medgemma_r4_alpha4_20251014_013202/merged_model",
        "is_base": False,
    },
    {
        "name": "r16_alpha16_epochs2",
        "path": "outputs_medgemma_brats/brats_medgemma_r16_alpha16_20251014_023559/merged_model",
        "is_base": False,
    }
]

# Simple test questions (text-only)
TEXT_QUESTIONS = [
    {"prompt": "Hello", "category": "Greeting"},
    {"prompt": "What is the capital of France?", "category": "General Knowledge"},
    {"prompt": "What is 2 + 2?", "category": "Math"},
    {"prompt": "Who are you?", "category": "Identity"},
    {"prompt": "What is a brain tumor?", "category": "Medical Knowledge"},
]

# Image questions - ALL IMAGES
IMAGE_QUESTIONS = [
    # Brain MRI scans
    {
        "prompt": "Analyze this brain MRI scan. What regions or structures can you identify?",
        "image": "images/single_slice.jpg",
        "category": "Brain MRI - Single Slice"
    },
    {
        "prompt": "Analyze this brain MRI scan. What regions or structures can you identify?",
        "image": "images/three_slices.jpg",
        "category": "Brain MRI - Three Slices"
    },
    {
        "prompt": "Analyze this brain MRI scan. What regions or structures can you identify?",
        "image": "images/five_slices.jpg",
        "category": "Brain MRI - Five Slices"
    },
    # General/Medical images
    {
        "prompt": "What do you see in this image? Describe it in detail.",
        "image": "images/image_1.png",
        "category": "Medical Image - Chest X-ray"
    },
    {
        "prompt": "What do you see in this image? Describe it in detail.",
        "image": "images/image_2.png",
        "category": "Medical Image - Brain Scan"
    },
    {
        "prompt": "What do you see in this image? Describe it in detail.",
        "image": "images/image_3.png",
        "category": "Medical Image - Tissue Sample"
    },
    {
        "prompt": "What do you see in this image? Describe it in detail.",
        "image": "images/image_4.jpg",
        "category": "Medical Image - Blood Cells"
    },
    {
        "prompt": "What do you see in this image? Describe it in detail.",
        "image": "images/image_5.jpg",
        "category": "Medical Image - Breast Mammogram"
    },
    {
        "prompt": "What do you see in this image? Describe it in detail.",
        "image": "images/image_6.jpg",
        "category": "Medical Image - Bone X-ray"
    },
    {
        "prompt": "What do you see in this image? Describe it in detail.",
        "image": "images/image_7.png",
        "category": "General Image - Puppies"
    },
    {
        "prompt": "What do you see in this image? Describe it in detail.",
        "image": "images/image_8.png",
        "category": "General Image - Eiffel Tower"
    },
]

# Generation parameters
MAX_NEW_TOKENS = 200
TEMPERATURE = 0.7
TOP_P = 0.9

OUTPUT_CSV = f"simple_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"


def clean_response(raw_response):
    """Extract only the actual response, removing chat template markup."""
    response = raw_response
    
    # Split by "model" marker and get last part
    if "\nmodel\n" in response:
        response = response.split("\nmodel\n")[-1]
    
    # Remove any remaining user sections
    if "\nuser\n" in response:
        response = response.split("\nuser\n")[0]
    
    # Clean special tokens
    response = response.replace("<|im_start|>", "").replace("<|im_end|>", "")
    response = response.replace("assistant", "").replace("model", "")
    
    return response.strip()


def load_model(model_config):
    """Load a model."""
    print(f"\n{'='*70}")
    print(f"Loading: {model_config['name']}")
    print(f"{'='*70}")
    
    model, tokenizer = FastVisionModel.from_pretrained(
        model_config['path'],
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
    )
    
    tokenizer = get_chat_template(tokenizer, chat_template="gemma-3")
    FastVisionModel.for_inference(model)
    
    print(f"✓ Loaded!")
    return model, tokenizer


def generate_response(model, tokenizer, prompt, image_path=None):
    """Generate a clean response from the model."""
    try:
        # Prepare message
        if image_path and os.path.exists(image_path):
            image = Image.open(image_path)
            messages = [
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image", "image": image}
                ]}
            ]
        else:
            messages = [{"role": "user", "content": prompt}]
        
        # Tokenize
        input_text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        
        if image_path and os.path.exists(image_path):
            inputs = tokenizer(
                text=input_text,
                images=[image],
                add_special_tokens=False,
                return_tensors="pt",
            ).to(model.device)
        else:
            inputs = tokenizer(
                input_text,
                add_special_tokens=False,
                return_tensors="pt",
            ).to(model.device)
        
        # Generate
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            do_sample=True,
            use_cache=True,
        )
        
        # Decode and clean
        raw_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        clean = clean_response(raw_response)
        
        return clean
    
    except Exception as e:
        return f"ERROR: {str(e)}"


def run_tests():
    """Run all tests and save results."""
    print("="*70)
    print("SIMPLE MODEL COMPARISON TEST")
    print("="*70)
    print(f"Testing {len(MODELS_TO_TEST)} models")
    print(f"Text questions: {len(TEXT_QUESTIONS)}")
    print(f"Image questions: {len(IMAGE_QUESTIONS)}")
    print(f"Output: {OUTPUT_CSV}")
    print("="*70)
    
    results = []
    
    # Test each model
    for model_config in MODELS_TO_TEST:
        model, tokenizer = load_model(model_config)
        
        # Test text questions
        print(f"\nTesting text questions...")
        for i, q in enumerate(TEXT_QUESTIONS, 1):
            print(f"  [{i}/{len(TEXT_QUESTIONS)}] {q['category'][:40]}...", end=" ")
            
            import time
            start = time.time()
            response = generate_response(model, tokenizer, q['prompt'])
            gen_time = time.time() - start
            
            results.append({
                'model': model_config['name'],
                'type': 'text',
                'category': q['category'],
                'prompt': q['prompt'],
                'image': '',
                'response': response,
                'length': len(response),
                'time': round(gen_time, 2)
            })
            
            print(f"✓ ({gen_time:.1f}s)")
        
        # Test image questions
        print(f"\nTesting image questions...")
        for i, q in enumerate(IMAGE_QUESTIONS, 1):
            print(f"  [{i}/{len(IMAGE_QUESTIONS)}] {q['category'][:40]}...", end=" ")
            
            start = time.time()
            response = generate_response(model, tokenizer, q['prompt'], q['image'])
            gen_time = time.time() - start
            
            results.append({
                'model': model_config['name'],
                'type': 'image',
                'category': q['category'],
                'prompt': q['prompt'],
                'image': q['image'],
                'response': response,
                'length': len(response),
                'time': round(gen_time, 2)
            })
            
            print(f"✓ ({gen_time:.1f}s)")
        
        # Clear memory
        del model, tokenizer
        torch.cuda.empty_cache()
    
    # Save to CSV
    print(f"\n{'='*70}")
    print("SAVING RESULTS")
    print(f"{'='*70}")
    
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['model', 'type', 'category', 'prompt', 'image', 'response', 'length', 'time'], delimiter='|')
        writer.writeheader()
        writer.writerows(results)
    
    print(f"✓ Saved: {OUTPUT_CSV}")
    print(f"  Total responses: {len(results)}")
    print(f"  Delimiter: pipe (|)")
    print("="*70)


if __name__ == "__main__":
    run_tests()

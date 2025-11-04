# 🏥 Medical Image Classification with MobileViT and LoRA

## 📋 Overview

This project implements **LoRA fine-tuning** on **MobileViT-Small** for medical image classification across three datasets: Brain Tumor, Chest X-Ray, and Lung Cancer. Compare parameter-efficient fine-tuning vs full fine-tuning approaches.

## 🤖 Pre-trained Models

All models available on 🤗 **Hugging Face Hub**:

- 🧠 **Brain Tumor**: [Jesteban247/mobilevit_small-brain_tumor](https://huggingface.co/Jesteban247/mobilevit_small-brain_tumor)
- 🫁 **Chest X-Ray**: [Jesteban247/mobilevit_small-chest_xray](https://huggingface.co/Jesteban247/mobilevit_small-chest_xray)
- 🧬 **Lung Cancer**: [Jesteban247/mobilevit_small-lung_cancer_histopathological](https://huggingface.co/Jesteban247/mobilevit_small-lung_cancer_histopathological)

## 📁 Project Structure

```
├── 🧠 EDA.ipynb              # Dataset download & preprocessing
├── 🏋️ Train.py               # Training script (LoRA + full)
├── 🧪 Experiments.py         # Grid search experiments
├── 📊 Analysis.ipynb         # Results analysis
├── 🔍 Prediction.ipynb       # Inference + Grad-CAM
├── 🔍 Test_cam_methods.ipynb # Testing different CAM methods
├── 📋 requirements.txt       # Dependencies
├── 📁 Images/                # Dataset and visualization images
└── 📁 outputs/               # Training logs
```

## 🚀 Quick Start

### 1. 🛠️ Environment Setup

```bash
# Install Miniconda
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# Initialize Conda
source ~/miniconda3/bin/activate

# Accept Terms of Service
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Create and activate environment
conda create -n medical-vision python=3.12 -y
conda activate medical-vision
pip install -r requirements.txt

# Create Downloads Folder
mkdir -p Downloads
```

### 2. ⚡ Accelerate Configuration (2 GPUs)

This project is intended to run on a single machine with 2 GPUs. Run the interactive Accelerate setup and choose the options below when prompted. You can also create or edit an `accelerate` config file manually.

Interactive (recommended):

```bash
accelerate config
```

When prompted, choose these options (recommended):

- **Compute Environment**: This machine - Running on local hardware
- **Machine Type**: multi-GPU - Multiple GPUs available
- **Number of Machines**: 1 - Single-node setup
- **Check Distributed Operations**: no - Skip error checking for speed (may miss timeouts)
- **Torch Dynamo Optimization**: no - Not using PyTorch's Dynamo compiler
- **DeepSpeed**: no - Not using DeepSpeed for memory optimization
- **FullyShardedDataParallel**: no - Not using FSDP for parameter sharding
- **Megatron-LM**: no - Not using Megatron for large language models
- **GPUs for Training**: n, all - Using all n GPUs
- **NUMA Efficiency**: no - For NVIDIA hardware performance
- **Mixed Precision**: fp16 - 16-bit floating point for faster training

### 3. 🧠 Explore EDA.ipynb

**Open and run `EDA.ipynb`** - This notebook:
- Downloads all datasets automatically
- Processes and organizes data (train/val/test splits)
- Creates `Data/` folder with processed datasets
- Generates config files

### 4. 🏋️ Check Train.py

Review the training script that supports both LoRA and full fine-tuning.

### 5. 🧪 Run Experiments

```bash
python Experiments.py
```

This runs grid search comparing LoRA vs full fine-tuning across learning rates.

### 6. 📊 Jump to Analysis.ipynb

**Open `Analysis.ipynb`** to analyze training results and compare performance.

### 7. 🔍 Try Prediction.ipynb

**Open `Prediction.ipynb`** for inference with Grad-CAM visualization.

## 📸 Example Predictions

Here are side-by-side examples of model predictions on sample images from each dataset:

| Brain Tumor | Chest X-Ray | Lung Histopathology |
|-------------|-------------|---------------------|
| ![Brain Tumor](Images/Brain_Tumor.png) | ![Chest X-Ray](Images/Chest_X-Ray.png) | ![Lung Histopathology](Images/Lung_Histopathology.png) |

## 🔬 Key Features

- ⚡ **LoRA Fine-tuning**: ~98% parameter reduction
- 🏥 **Medical Datasets**: Brain tumors, chest X-rays, lung cancer
- 🔥 **Grad-CAM**: Model explainability visualizations
- 🤗 **Hugging Face**: Pre-trained models ready to use

## ⚠️ Medical Disclaimer

This project is for research purposes only. Models should not be used for clinical diagnosis without proper validation.
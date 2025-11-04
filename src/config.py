"""Configuration for MedGemma AI servers and limits"""

import os

# ============================================================================
# ENVIRONMENT DETECTION
# ============================================================================
IS_HF_SPACE = os.getenv("SPACE_ID") is not None

# ============================================================================
# DEVICE DETECTION
# ============================================================================
# Delay torch import in HF Spaces (spaces must be imported first)
if not IS_HF_SPACE:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
else:
    # Will be set after spaces is imported
    DEVICE = None  # Set by server_hf.py after spaces import

# ============================================================================
# VLM MODE - CONDITIONAL (HF Spaces or Local GGUF)
# ============================================================================
if IS_HF_SPACE:
    # HF Spaces: Use transformers with preloaded models
    HF_BASE_MODEL = "unsloth/medgemma-4b-it"
    HF_FT_MODEL = "Jesteban247/brats_medgemma"
    MAX_CONCURRENT_USERS = 5  # Lower for HF Spaces
    MAX_NUM_IMAGES = int(os.getenv("MAX_NUM_IMAGES", "5"))
else:
    # Local: Use llama-cpp servers for vision language models
    BASE_SERVER_URL = "http://localhost:8080/v1/chat/completions"  # Base MedGemma model
    FT_SERVER_URL = "http://localhost:8081/v1/chat/completions"    # Fine-tuned BraTS model
    SERVER_URL = BASE_SERVER_URL  # Default to base model
    MAX_CONCURRENT_USERS = 10  # Concurrent chat requests
    REQUEST_TIMEOUT = 300  # Total request timeout (seconds)
    CONNECT_TIMEOUT = 10  # Connection timeout (seconds)

# Session Management
HISTORY_EXCHANGES = 3  # Number of exchanges to keep (3 exchanges = 6 messages)
SESSION_TTL = 300  # Session timeout in seconds (5 minutes)

# Medical AI Models Server
MODELS_SERVER_URL = "http://localhost:8000"
MODELS_SERVER_TIMEOUT = 300  # Model inference timeout (seconds)

# UI Configuration
IMAGE_PREVIEW_HEIGHT = 400  # Standard image preview height (px)
SEG3D_PREVIEW_HEIGHT = 700  # Segmentation image preview height (px)

# Segmentation Visualization
DOWNSAMPLE_FACTOR = 4  # 3D visualization downsample factor
MAX_BRAIN_POINTS = 15000  # Max brain tissue points in 3D view
MAX_TUMOR_POINTS = 3000  # Max tumor points per class in 3D view
SLICE_OFFSETS_3 = [-10, 0, 10]  # Slice offsets for 3-slice view
SLICE_OFFSETS_5 = [-10, -5, 0, 5, 10]  # Slice offsets for 5-slice view
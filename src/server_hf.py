"""HF Spaces VLM inference using transformers (spaces.GPU decorator)"""

import os
import re
from collections.abc import Iterator
from threading import Thread

import gradio as gr
import spaces
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText, TextIteratorStreamer

from .logger import setup_logger
from .config import HF_BASE_MODEL, HF_FT_MODEL, MAX_NUM_IMAGES

# Set DEVICE now that torch is imported after spaces
import src.config as config
if config.DEVICE is None:
    config.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logger = setup_logger(__name__)

# Preload both models at startup
models = {}
processors = {}

def load_model(model_id: str):
    """Load and cache a model"""
    if model_id in models:
        return
    logger.info(f"🔄 Preloading model: {model_id}")
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, device_map="auto", torch_dtype=torch.bfloat16
    )
    model.generation_config.do_sample = True
    models[model_id] = model
    processors[model_id] = processor
    logger.info(f"✅ Model loaded: {model_id}")

# Load both models at module import
logger.info("🚀 Initializing HF Spaces models...")
load_model(HF_BASE_MODEL)
load_model(HF_FT_MODEL)
logger.info("✅ All models preloaded")


def count_files_in_new_message(paths: list[str]) -> int:
    return len([path for path in paths if not path.endswith(".mp4")])


def count_files_in_history(history: list[dict]) -> int:
    image_count = 0
    for item in history:
        if item["role"] != "user" or isinstance(item["content"], str):
            continue
        image_count += 1
    return image_count


def validate_media_constraints(message: dict, history: list[dict]) -> bool:
    new_image_count = count_files_in_new_message(message["files"])
    history_image_count = count_files_in_history(history)
    image_count = history_image_count + new_image_count
    if image_count > MAX_NUM_IMAGES:
        gr.Warning(f"You can upload up to {MAX_NUM_IMAGES} images.")
        return False
    if "<image>" in message["text"] and message["text"].count("<image>") != new_image_count:
        gr.Warning("The number of <image> tags in the text does not match the number of images.")
        return False
    return True


def process_interleaved_images(message: dict) -> list[dict]:
    logger.debug(f"{message['files']=}")
    parts = re.split(r"(<image>)", message["text"])
    logger.debug(f"{parts=}")

    content = []
    image_index = 0
    for part in parts:
        logger.debug(f"{part=}")
        if part == "<image>":
            content.append({"type": "image", "url": message["files"][image_index]})
            logger.debug(f"file: {message['files'][image_index]}")
            image_index += 1
        elif part.strip():
            content.append({"type": "text", "text": part.strip()})
        elif isinstance(part, str) and part != "<image>":
            content.append({"type": "text", "text": part})
    logger.debug(f"{content=}")
    return content


def process_new_user_message(message: dict) -> list[dict]:
    if not message["files"]:
        return [{"type": "text", "text": message["text"]}]

    if "<image>" in message["text"]:
        return process_interleaved_images(message)

    return [
        {"type": "text", "text": message["text"]},
        *[{"type": "image", "url": path} for path in message["files"]],
    ]


def process_history(history: list[dict]) -> list[dict]:
    messages = []
    current_user_content: list[dict] = []
    for item in history:
        if item["role"] == "assistant":
            if current_user_content:
                messages.append({"role": "user", "content": current_user_content})
                current_user_content = []
            messages.append({"role": "assistant", "content": [{"type": "text", "text": item["content"]}]})
        else:
            content = item["content"]
            if isinstance(content, str):
                current_user_content.append({"type": "text", "text": content})
            else:
                current_user_content.append({"type": "image", "url": content[0]})
    if current_user_content:
        messages.append({"role": "user", "content": current_user_content})
    return messages


@spaces.GPU(duration=120)
def run_hf_inference(message: dict, history: list[dict], system_prompt: str = "", max_new_tokens: int = 2048, model_id: str = None) -> Iterator[str]:
    """
    HF Spaces inference with @spaces.GPU decorator
    
    Args:
        message: Current message dict with 'text' and optional 'files'
        history: Gradio chat history
        system_prompt: System message
        max_new_tokens: Max tokens to generate
        model_id: Model identifier (HF_BASE_MODEL or HF_FT_MODEL)
    """
    # Default to base model if not specified
    if model_id is None:
        model_id = HF_BASE_MODEL
    
    if model_id not in models:
        raise ValueError(f"Model {model_id} not preloaded.")

    model = models[model_id]
    processor = processors[model_id]
    model_name = "Base" if model_id == HF_BASE_MODEL else "FT"
    logger.info(f"🤖 Using model: {model_name} ({model_id})")

    if not validate_media_constraints(message, history):
        yield ""
        return

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
    messages.extend(process_history(history))
    messages.append({"role": "user", "content": process_new_user_message(message)})

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device=model.device, dtype=torch.bfloat16)

    streamer = TextIteratorStreamer(processor, timeout=30.0, skip_prompt=True, skip_special_tokens=True)
    generate_kwargs = dict(
        inputs,
        max_new_tokens=max_new_tokens,
        streamer=streamer,
        do_sample=True,
        temperature=1.0,
        top_p=0.95,
        top_k=64,
        min_p=0.0,
    )
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    output = ""
    for delta in streamer:
        output += delta
        yield output
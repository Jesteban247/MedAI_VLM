"""Medical AI models client"""

import aiohttp
from typing import Optional, Dict, Any
from .logger import setup_logger
from .config import MODELS_SERVER_URL, MODELS_SERVER_TIMEOUT

logger = setup_logger(__name__)


async def classify_image(image_path: str, model: str) -> Optional[Dict[str, Any]]:
    """Classify medical image"""
    try:
        logger.info(f"🔬 Classification | Model: {model} | Image: {image_path.split('/')[-1]}")
        timeout = aiohttp.ClientTimeout(total=MODELS_SERVER_TIMEOUT)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            payload = {"image_path": image_path, "model": model}
            async with session.post(f"{MODELS_SERVER_URL}/classify", json=payload) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    top_pred = result['top_prediction']
                    logger.info(f"✅ Classification | {model} → {top_pred['class_name']} ({top_pred['confidence']*100:.1f}%)")
                    return result
                else:
                    error = await resp.text()
                    logger.error(f"❌ Classification failed: {resp.status} - {error}")
                    return None
    except Exception as e:
        logger.error(f"❌ Classification error: {e}")
        return None

async def detect_objects(image_path: str, model: str) -> Optional[Dict[str, Any]]:
    """Detect objects in medical image"""
    try:
        logger.info(f"🔍 Detection | Model: {model} | Image: {image_path.split('/')[-1]}")
        timeout = aiohttp.ClientTimeout(total=MODELS_SERVER_TIMEOUT)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            payload = {"image_path": image_path, "model": model}
            async with session.post(f"{MODELS_SERVER_URL}/detect", json=payload) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    logger.info(f"✅ Detection | {model} → {result['total_detections']} objects found")
                    return result
                else:
                    error = await resp.text()
                    logger.error(f"❌ Detection failed: {resp.status} - {error}")
                    return None
    except Exception as e:
        logger.error(f"❌ Detection error: {e}")
        return None

async def segment_case(case_path: str, model: str = "brats") -> Optional[Dict[str, Any]]:
    """Segment 3D brain tumor case"""
    try:
        case_name = case_path.split('/')[-1]
        logger.info(f"📊 Segmentation | Model: {model} | Case: {case_name}")
        timeout = aiohttp.ClientTimeout(total=MODELS_SERVER_TIMEOUT)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            payload = {"case_path": case_path, "model": model}
            async with session.post(f"{MODELS_SERVER_URL}/segment", json=payload) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    logger.info(f"✅ Segmentation | {model} → Avg Dice: {result['average_dice']:.3f}")
                    return result
                else:
                    error = await resp.text()
                    logger.error(f"❌ Segmentation failed: {resp.status} - {error}")
                    return None
    except Exception as e:
        logger.error(f"❌ Segmentation error: {e}")
        return None

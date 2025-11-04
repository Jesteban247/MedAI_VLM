"""VLM client with file-based session history - supports both local GGUF and HF Spaces"""

import aiohttp
import asyncio
import json
import base64
import os
from typing import Optional, Dict, Any, AsyncGenerator
from .logger import setup_logger
from .config import IS_HF_SPACE, MAX_CONCURRENT_USERS
from .session_manager import session_manager

# Conditional imports based on environment
if not IS_HF_SPACE:
    from .config import SERVER_URL, REQUEST_TIMEOUT, CONNECT_TIMEOUT

SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT_USERS)
logger = setup_logger(__name__)

if IS_HF_SPACE:
    logger.info("🌐 Running in HF Spaces mode (transformers)")
else:
    logger.info("💻 Running in local GGUF mode (llama-cpp)")

def get_active_sessions() -> list:
    """Get list of currently active sessions"""
    return session_manager.get_active_sessions()

def cleanup_expired_sessions() -> int:
    """Cleanup sessions that exceeded TTL"""
    return session_manager.cleanup_expired_sessions()

async def _encode_image(path: str) -> tuple[str, str]:
    """
    Encode image to base64 and detect format
    Returns: (base64_string, mime_type)
    """
    def _encode():
        if not os.path.exists(path):
            logger.warning(f"Image not found: {path}")
            return None, None
            
        with open(path, "rb") as f:
            img_bytes = f.read()
        
        # Detect image format from magic bytes
        if img_bytes.startswith(b'\x89PNG'):
            mime_type = "image/png"
        elif img_bytes.startswith(b'\xFF\xD8\xFF'):
            mime_type = "image/jpeg"
        elif img_bytes.startswith(b'GIF'):
            mime_type = "image/gif"
        elif img_bytes.startswith(b'RIFF') and b'WEBP' in img_bytes[:12]:
            mime_type = "image/webp"
        else:
            mime_type = "image/jpeg"  # Default fallback
        
        return base64.b64encode(img_bytes).decode(), mime_type
    
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _encode)


def _extract_text_from_content(content: Any) -> str:
    """Extract text from message content (handles string/dict/list formats)"""
    if isinstance(content, str):
        return content
    elif isinstance(content, dict):
        return content.get("text", "")
    elif isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                return item.get("text", "")
    return ""


def _find_image_in_context_window(all_messages: list, max_messages: int, session_id: str) -> Optional[str]:
    """
    Find most recent image that will remain in context window after adding new messages
    
    Args:
        all_messages: Current message history
        max_messages: Maximum messages to keep in window
        session_id: Session identifier for logging
        
    Returns:
        Path to image file or None
    """
    # Calculate which messages will remain after adding current message + assistant response
    if len(all_messages) >= max_messages - 1:
        # Keep only messages that will survive: drop oldest 2 positions
        messages_that_will_remain = all_messages[2:] if len(all_messages) >= 2 else all_messages
    else:
        messages_that_will_remain = all_messages
    
    # Find most recent image in messages that will remain
    for msg in reversed(messages_that_will_remain):
        img_path = msg.get("image_path")
        if img_path and os.path.exists(img_path):
            logger.info(f"🔍 Session {session_id[:8]} | Found image in context window")
            return img_path
    
    # Check if there was an image that got dropped
    if len(all_messages) >= max_messages - 1:
        for msg in all_messages[:2]:  # Check dropped messages
            if msg.get("image_path"):
                logger.info(f"🗑️  Session {session_id[:8]} | Image outside context window (will be deleted)")
    
    return None


def _build_history_messages(session_id: str) -> list[Dict[str, Any]]:
    """Build API message history from session files"""
    messages = session_manager.get_messages_for_context(session_id)
    return [{"role": msg["role"], "content": msg["content"]} for msg in messages]


def _find_most_recent_image(session_id: str) -> Optional[str]:
    """Find most recent image in session history"""
    messages = session_manager.get_messages_for_context(session_id)
    
    for msg in reversed(messages):
        if msg.get("image_path") and os.path.exists(msg["image_path"]):
            return msg["image_path"]
    
    return None

async def respond_stream_hf(message, history, system_message: str, max_tokens: int, temperature: float, top_p: float, session_id: Optional[str] = None, model_choice: str = "Base (General)") -> AsyncGenerator[str, None]:
    """
    HF Spaces streaming response using transformers
    
    Args:
        message: Current user message (text or dict with text/files)
        history: Gradio history
        system_message: System prompt
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        session_id: Optional session identifier
        model_choice: Model selection string
    """
    from .config import HF_BASE_MODEL, HF_FT_MODEL
    from .server_hf import run_hf_inference
    
    # Map model choice to model ID
    model_id = HF_FT_MODEL if model_choice == "Fine-Tuned (BraTS)" else HF_BASE_MODEL
    
    # Generate session ID if not provided
    if session_id is None:
        session_id = str(id(history)) if history is not None else "default"
    
    # Log request
    model_type = "FT" if model_choice == "Fine-Tuned (BraTS)" else "Base"
    has_image = isinstance(message, dict) and message.get("files")
    img_status = "with image" if has_image else "text only"
    logger.info(f"🤖 HF Spaces | Session: {session_id[:8]} | Model: {model_type} | {img_status}")
    
    # Convert Gradio history to HF format
    hf_history = []
    for item in history:
        role = item.get("role", "user")
        content = item.get("content", "")
        
        # Handle multimodal content
        if isinstance(content, list):
            # Already in multimodal format
            hf_history.append({"role": role, "content": content})
        elif isinstance(content, str):
            hf_history.append({"role": role, "content": content})
        else:
            # Image tuple format from Gradio
            hf_history.append({"role": role, "content": content})
    
    # Stream response from HF inference (sync generator in async context)
    try:
        # Run sync generator in executor to avoid blocking
        loop = asyncio.get_event_loop()
        gen = run_hf_inference(message, hf_history, system_message, max_tokens, model_id)
        
        # Yield chunks as they come
        def get_next(g):
            try:
                return next(g), False
            except StopIteration:
                return None, True
        
        while True:
            chunk, done = await loop.run_in_executor(None, get_next, gen)
            if done:
                break
            if chunk is not None:
                yield chunk
                # Small delay to allow UI to update
                await asyncio.sleep(0.01)
                
    except Exception as e:
        logger.error(f"❌ HF inference error: {e}")
        yield f"Error: {str(e)}"


async def respond_stream(message, history, system_message: str, max_tokens: int, temperature: float, top_p: float, session_id: Optional[str] = None, server_url: Optional[str] = None, model_choice: str = "Base (General)") -> AsyncGenerator[str, None]:
    """
    Stream VLM response - routes to HF Spaces or local GGUF based on environment
    
    Args:
        message: Current user message (text or dict with text/files)
        history: Gradio history (used to extract session info)
        system_message: System prompt
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        session_id: Optional session identifier
        server_url: Optional custom server URL (local only)
        model_choice: Model selection string
    """
    # Route to appropriate inference method
    if IS_HF_SPACE:
        async for chunk in respond_stream_hf(message, history, system_message, max_tokens, temperature, top_p, session_id, model_choice):
            yield chunk
        return
    
    # Local GGUF inference below
    # Cleanup expired sessions periodically
    cleanup_expired_sessions()
    
    # Generate session ID from history object if not provided
    if session_id is None:
        session_id = str(id(history)) if history is not None else "default"
    
    # Ensure session exists
    if not session_manager.session_exists(session_id):
        session_manager.create_session(session_id)
    
    session_manager.update_activity(session_id)
    active_count = len(get_active_sessions())
    
    history_msgs = session_manager.get_session_history(session_id)
    
    # Extract text from current message
    prompt = _extract_text_from_content(message.get("text") if isinstance(message, dict) else message)
    
    # Handle new image upload
    current_image_path = None
    has_image = isinstance(message, dict) and message.get("files") and len(message["files"]) > 0
    if has_image:
        current_image_path = message["files"][0]
    
    # Build conversation history from session files
    all_messages = session_manager.get_messages_for_context(session_id)
    
    # Log request summary
    img_status = "with image" if has_image else "text only"
    logger.info(f"Request | Session: {session_id[:8]} | {img_status} | History: {len(history_msgs)} msgs")
    
    # Determine which image to use (new upload or cached from history)
    from .config import HISTORY_EXCHANGES
    max_messages = HISTORY_EXCHANGES * 2
    
    if current_image_path:
        image_to_use = current_image_path
    else:
        image_to_use = _find_image_in_context_window(all_messages, max_messages, session_id)
    
    # Build API messages from history
    messages = [{"role": msg["role"], "content": msg["content"]} for msg in all_messages]
    
    # Build current message with or without image
    if image_to_use:
        image_b64, mime_type = await _encode_image(image_to_use)
        if image_b64:
            messages.append({
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_b64}"}},
                    {"type": "text", "text": prompt}
                ]
            })
            if current_image_path:
                logger.info(f"📤 Sending WITH NEW IMAGE ({mime_type})")
            else:
                logger.info(f"📤 Sending WITH IMAGE from history ({mime_type})")
        else:
            messages.append({"role": "user", "content": prompt})
            logger.info(f"📤 TEXT ONLY (image load failed)")
    else:
        messages.append({"role": "user", "content": prompt})
        logger.info(f"📤 TEXT ONLY (no image in window)")
    
    # Count images in payload
    image_count = sum(1 for msg in messages if isinstance(msg.get("content"), list))
    logger.info(f"📊 Sending: {len(messages)} msgs | {image_count} image(s)")
    
    payload = {
        "model": "medgemma",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": True
    }
    
    # Use custom server URL if provided, otherwise use default
    target_url = server_url if server_url else SERVER_URL
    
    try:
        timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT, connect=CONNECT_TIMEOUT)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with SEMAPHORE:
                async with session.post(target_url, json=payload) as resp:
                    if resp.status != 200:
                        logger.error(f"❌ Server error: {resp.status}")
                        yield f"Error: {resp.status}"
                        return
                    
                    full, token_count = "", 0
                    # Stream response chunks
                    async for line in resp.content:
                        text = line.decode().strip()
                        if not text.startswith('data: '):
                            continue
                        
                        data = text[6:]  # Remove 'data: ' prefix
                        if data == '[DONE]':
                            break
                        
                        try:
                            obj = json.loads(data)
                            chunk = obj.get('choices', [{}])[0].get('delta', {}).get('content', '')
                            if chunk:
                                full += chunk
                                token_count += 1
                                yield full
                        except:
                            continue  # Skip malformed chunks
                    
                    # Handle empty response (context exceeded)
                    if token_count == 0:
                        error_msg = "⚠️ Context size exceeded. Please start a new conversation or reduce history."
                        logger.warning(f"⚠️  Session {session_id[:8]} | Context exceeded")
                        yield error_msg
                    else:
                        logger.info(f"✅ COMPLETED | Session: {session_id[:8]} | Tokens: {token_count} | Active: {len(get_active_sessions())}")
                        
                        # Save messages to session history
                        session_manager.add_message(session_id, "user", prompt, current_image_path)
                        session_manager.add_message(session_id, "assistant", full, None)
    except Exception as e:
        logger.error(f"❌ Exception: {e}")
        yield f"Error: {e}"
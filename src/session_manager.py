"""File-based session management with automatic cleanup"""

import os
import json
import shutil
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
from .logger import setup_logger
from .config import HISTORY_EXCHANGES, SESSION_TTL

logger = setup_logger(__name__)

# Configuration
HISTORY_DIR = Path("History")

class SessionManager:
    """Manages session history with file-based storage"""
    
    def __init__(self):
        """Initialize session manager and create History directory"""
        HISTORY_DIR.mkdir(exist_ok=True)
        logger.info(f"Session manager initialized | History: {HISTORY_DIR}")
    
    def _get_session_dir(self, session_id: str) -> Path:
        """Get session directory path"""
        return HISTORY_DIR / f"Session_{session_id}"
    
    def _get_session_json(self, session_id: str) -> Path:
        """Get session JSON file path"""
        return self._get_session_dir(session_id) / "history.json"
    
    def _get_images_dir(self, session_id: str) -> Path:
        """Get session images directory path"""
        return self._get_session_dir(session_id) / "images"
    
    def create_session(self, session_id: str) -> None:
        """Create new session directory structure"""
        session_dir = self._get_session_dir(session_id)
        images_dir = self._get_images_dir(session_id)
        
        session_dir.mkdir(exist_ok=True)
        images_dir.mkdir(exist_ok=True)
        
        # Initialize history JSON
        now = datetime.now().isoformat()
        history_data = {
            "session_id": session_id,
            "created_at": now,
            "last_activity": now,
            "messages": []
        }
        
        with open(self._get_session_json(session_id), 'w') as f:
            json.dump(history_data, f, indent=2)
        
        logger.info(f"🟢 NEW SESSION: {session_id[:8]}")
    
    def session_exists(self, session_id: str) -> bool:
        """Check if session exists"""
        return self._get_session_dir(session_id).exists()
    
    def get_session_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Load session history from JSON"""
        if not self.session_exists(session_id):
            self.create_session(session_id)
            return []
        
        json_path = self._get_session_json(session_id)
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        return data.get("messages", [])
    
    def add_message(self, session_id: str, role: str, content: str, 
                   image_path: Optional[str] = None) -> None:
        """
        Add message to session history with sliding window (3 exchanges)
        
        Args:
            session_id: Session identifier
            role: 'user' or 'assistant'
            content: Message text
            image_path: Optional path to image file
        """
        if not self.session_exists(session_id):
            self.create_session(session_id)
        
        # Load current history
        json_path = self._get_session_json(session_id)
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Prepare and add message
        message = self._create_message(session_id, role, content, image_path, len(data['messages']))
        data["messages"].append(message)
        data["last_activity"] = datetime.now().isoformat()
        
        # Trim to keep only last 3 exchanges (6 messages)
        self._trim_history(session_id, data)
        
        # Save updated history
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _create_message(self, session_id: str, role: str, content: str, 
                       image_path: Optional[str], msg_count: int) -> Dict[str, Any]:
        """Create a message dict with optional image"""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "image": None
        }
        
        # Handle image if provided
        if image_path and os.path.exists(image_path):
            image_filename = f"msg_{msg_count}_{int(time.time())}.png"
            dest_path = self._get_images_dir(session_id) / image_filename
            shutil.copy2(image_path, dest_path)
            message["image"] = image_filename
        
        return message
    
    def _trim_history(self, session_id: str, data: Dict[str, Any]) -> None:
        """Trim message history to max exchanges and delete old images"""
        max_messages = HISTORY_EXCHANGES * 2
        
        if len(data["messages"]) <= max_messages:
            return
        
        # Calculate how many to remove
        num_to_remove = len(data["messages"]) - max_messages
        removed = data["messages"][:num_to_remove]
        data["messages"] = data["messages"][num_to_remove:]
        
        # Delete old images
        for msg in removed:
            if msg.get("image"):
                old_img = self._get_images_dir(session_id) / msg["image"]
                if old_img.exists():
                    old_img.unlink()
                    logger.info(f"🗑️  Image dropped: {msg['image']}")
    
    def get_messages_for_context(self, session_id: str) -> List[Dict[str, Any]]:
        """Get messages for conversation context with full image paths"""
        messages = self.get_session_history(session_id)
        recent = messages  # Already trimmed to 3 exchanges in add_message
        
        # Convert image filenames to full paths
        for msg in recent:
            if msg.get("image"):
                msg["image_path"] = str(self._get_images_dir(session_id) / msg["image"])
        
        return recent
    
    def update_activity(self, session_id: str) -> None:
        """Update session last activity timestamp"""
        if not self.session_exists(session_id):
            return
        
        json_path = self._get_session_json(session_id)
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        data["last_activity"] = datetime.now().isoformat()
        
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def clear_session(self, session_id: str) -> None:
        """Clear session history (keep structure)"""
        if not self.session_exists(session_id):
            return
        
        # Clear images
        images_dir = self._get_images_dir(session_id)
        for img in images_dir.glob("*"):
            img.unlink()
        
        # Reset history
        json_path = self._get_session_json(session_id)
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        data["messages"] = []
        data["last_activity"] = datetime.now().isoformat()
        
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def delete_session(self, session_id: str) -> None:
        """Delete entire session directory"""
        session_dir = self._get_session_dir(session_id)
        if session_dir.exists():
            shutil.rmtree(session_dir)
            logger.info(f"🔴 EXPIRED: {session_id[:8]}")
    
    def cleanup_expired_sessions(self) -> int:
        """
        Delete sessions that exceeded TTL (5 minutes)
        
        Returns number of sessions deleted
        """
        if not HISTORY_DIR.exists():
            return 0
        
        current_time = datetime.now()
        deleted_count = 0
        
        for session_dir in HISTORY_DIR.glob("Session_*"):
            if self._should_delete_session(session_dir, current_time):
                deleted_count += 1
        
        return deleted_count
    
    def _should_delete_session(self, session_dir: Path, current_time: datetime) -> bool:
        """Check if session should be deleted and delete if expired"""
        json_path = session_dir / "history.json"
        
        if not json_path.exists():
            return False
        
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            last_activity = datetime.fromisoformat(data["last_activity"])
            idle_seconds = (current_time - last_activity).total_seconds()
            
            if idle_seconds > SESSION_TTL:
                session_id = data["session_id"]
                self.delete_session(session_id)
                return True
        
        except Exception as e:
            logger.error(f"Error checking session {session_dir.name}: {e}")
        
        return False
    
    def get_active_sessions(self) -> List[Dict[str, Any]]:
        """Get list of active sessions (within TTL)"""
        if not HISTORY_DIR.exists():
            return []
        
        current_time = datetime.now()
        active = []
        
        for session_dir in HISTORY_DIR.glob("Session_*"):
            json_path = session_dir / "history.json"
            
            if not json_path.exists():
                continue
            
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                
                last_activity = datetime.fromisoformat(data["last_activity"])
                idle_seconds = (current_time - last_activity).total_seconds()
                
                if idle_seconds <= SESSION_TTL:
                    active.append({
                        "session_id": data["session_id"][:8],
                        "created": datetime.fromisoformat(data["created_at"]).strftime("%H:%M:%S"),
                        "last_active": last_activity.strftime("%H:%M:%S"),
                        "messages": len(data["messages"]),
                        "idle_seconds": int(idle_seconds)
                    })
            
            except Exception as e:
                logger.error(f"Error reading session {session_dir.name}: {e}")
        
        return active

# Global instance
session_manager = SessionManager()

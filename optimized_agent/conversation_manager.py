"""
Conversation management system for persistent chat history and context tracking
"""
import json
import time
import re
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta

from config import config, EntityPatterns

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """Individual message in a conversation"""
    role: str
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        return cls(**data)


@dataclass
class ConversationContext:
    """Enhanced conversation context with persistence"""
    conversation_id: str
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    
    # Entity tracking
    current_part_number: Optional[str] = None
    current_model_number: Optional[str] = None
    mentioned_parts: List[str] = field(default_factory=list)
    mentioned_models: List[str] = field(default_factory=list)
    
    # Message history
    messages: List[Message] = field(default_factory=list)
    
    # Session metadata
    session_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a message to the conversation"""
        message = Message(
            role=role,
            content=content,
            metadata=metadata or {}
        )
        self.messages.append(message)
        self.last_updated = time.time()
        
        # Extract entities from user messages
        if role == "user":
            self._extract_entities(content)
    
    def _extract_entities(self, text: str) -> None:
        """Extract part and model numbers from text using enhanced patterns"""
        # Extract part numbers
        for pattern in EntityPatterns.PART_NUMBER_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                match = match.upper()  # Normalize to uppercase
                if match not in self.mentioned_parts:
                    self.mentioned_parts.append(match)
                    logger.info(f"Extracted part number: {match}")
                self.current_part_number = match
        
        # Extract model numbers
        for pattern in EntityPatterns.MODEL_NUMBER_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                match = match.upper()  # Normalize to uppercase
                if match not in self.mentioned_models:
                    self.mentioned_models.append(match)
                    logger.info(f"Extracted model number: {match}")
                self.current_model_number = match
    
    def get_context_summary(self) -> str:
        """Generate a context summary for the LLM"""
        summary_parts = []
        
        if self.current_part_number:
            summary_parts.append(f"Current part: {self.current_part_number}")
        
        if self.current_model_number:
            summary_parts.append(f"Current model: {self.current_model_number}")
        
        if self.mentioned_parts:
            recent_parts = self.mentioned_parts[-3:]
            summary_parts.append(f"Recent parts: {', '.join(recent_parts)}")
        
        if self.mentioned_models:
            recent_models = self.mentioned_models[-3:]
            summary_parts.append(f"Recent models: {', '.join(recent_models)}")
        
        return " | ".join(summary_parts) if summary_parts else "No context available"
    
    def get_recent_messages(self, count: int = None) -> List[Message]:
        """Get recent messages for context"""
        if count is None:
            count = config.conversation.CONTEXT_WINDOW_SIZE
        return self.messages[-count:] if self.messages else []
    
    def format_conversation_history(self, count: int = None) -> str:
        """Format recent conversation history for LLM prompt"""
        recent_messages = self.get_recent_messages(count)
        formatted = []
        
        for message in recent_messages:
            role = "User" if message.role == "user" else "Assistant"
            formatted.append(f"{role}: {message.content}")
        
        return "\n".join(formatted)
    
    def resolve_references(self, text: str) -> str:
        """Resolve references like 'this part', 'my model' using context"""
        resolved_text = text
        
        # Handle part references
        if self.current_part_number:
            part_patterns = [
                r'\b(this part|that part|the part|it)\b',
                r'\b(this item|that item|the item)\b'
            ]
            for pattern in part_patterns:
                resolved_text = re.sub(
                    pattern, 
                    self.current_part_number, 
                    resolved_text, 
                    flags=re.IGNORECASE
                )
        
        # Handle model references
        if self.current_model_number:
            model_patterns = [
                r'\b(my model|this model|the model|my appliance)\b',
                r'\b(my dishwasher|my refrigerator|my fridge)\b'
            ]
            for pattern in model_patterns:
                resolved_text = re.sub(
                    pattern, 
                    self.current_model_number, 
                    resolved_text, 
                    flags=re.IGNORECASE
                )
        
        return resolved_text
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'conversation_id': self.conversation_id,
            'created_at': self.created_at,
            'last_updated': self.last_updated,
            'current_part_number': self.current_part_number,
            'current_model_number': self.current_model_number,
            'mentioned_parts': self.mentioned_parts,
            'mentioned_models': self.mentioned_models,
            'messages': [msg.to_dict() for msg in self.messages],
            'session_metadata': self.session_metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationContext':
        """Create from dictionary"""
        messages = [Message.from_dict(msg_data) for msg_data in data.get('messages', [])]
        
        context = cls(
            conversation_id=data['conversation_id'],
            created_at=data.get('created_at', time.time()),
            last_updated=data.get('last_updated', time.time()),
            current_part_number=data.get('current_part_number'),
            current_model_number=data.get('current_model_number'),
            mentioned_parts=data.get('mentioned_parts', []),
            mentioned_models=data.get('mentioned_models', []),
            session_metadata=data.get('session_metadata', {})
        )
        context.messages = messages
        return context


class ConversationManager:
    """Manages conversation persistence and retrieval"""
    
    def __init__(self, storage_dir: str = None):
        self.storage_dir = Path(storage_dir or config.database.CONVERSATION_HISTORY_DIR)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory cache for active conversations
        self._active_conversations: Dict[str, ConversationContext] = {}
        self._last_cleanup = time.time()
    
    def get_conversation_file_path(self, conversation_id: str) -> Path:
        """Get file path for conversation storage"""
        return self.storage_dir / f"{conversation_id}.json"
    
    def get_or_create_conversation(self, conversation_id: str) -> ConversationContext:
        """Get existing conversation or create new one"""
        # Check memory cache first
        if conversation_id in self._active_conversations:
            return self._active_conversations[conversation_id]
        
        # Try to load from disk
        conversation = self.load_conversation(conversation_id)
        if conversation:
            self._active_conversations[conversation_id] = conversation
            return conversation
        
        # Create new conversation
        conversation = ConversationContext(conversation_id=conversation_id)
        self._active_conversations[conversation_id] = conversation
        logger.info(f"Created new conversation: {conversation_id}")
        return conversation
    
    def save_conversation(self, conversation: ConversationContext) -> bool:
        """Save conversation to disk"""
        try:
            file_path = self.get_conversation_file_path(conversation.conversation_id)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(conversation.to_dict(), f, indent=2, ensure_ascii=False)
            
            logger.debug(f"Saved conversation: {conversation.conversation_id}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to save conversation {conversation.conversation_id}: {e}")
            return False
    
    def load_conversation(self, conversation_id: str) -> Optional[ConversationContext]:
        """Load conversation from disk"""
        try:
            file_path = self.get_conversation_file_path(conversation_id)
            
            if not file_path.exists():
                return None
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            conversation = ConversationContext.from_dict(data)
            logger.debug(f"Loaded conversation: {conversation_id}")
            return conversation
        
        except Exception as e:
            logger.error(f"Failed to load conversation {conversation_id}: {e}")
            return None
    
    def add_message(self, conversation_id: str, role: str, content: str, 
                   metadata: Optional[Dict[str, Any]] = None, 
                   auto_save: bool = True) -> ConversationContext:
        """Add message to conversation with optional auto-save"""
        conversation = self.get_or_create_conversation(conversation_id)
        conversation.add_message(role, content, metadata)
        
        # Auto-save based on configuration
        if auto_save and len(conversation.messages) % config.conversation.AUTO_SAVE_INTERVAL == 0:
            self.save_conversation(conversation)
        
        return conversation
    
    def get_conversation_history(self, conversation_id: str, 
                               limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get conversation history as list of dictionaries"""
        conversation = self.get_or_create_conversation(conversation_id)
        messages = conversation.get_recent_messages(limit)
        return [msg.to_dict() for msg in messages]
    
    def clear_conversation(self, conversation_id: str) -> bool:
        """Clear conversation from memory and disk"""
        try:
            # Remove from memory
            if conversation_id in self._active_conversations:
                del self._active_conversations[conversation_id]
            
            # Remove from disk
            file_path = self.get_conversation_file_path(conversation_id)
            if file_path.exists():
                file_path.unlink()
            
            logger.info(f"Cleared conversation: {conversation_id}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to clear conversation {conversation_id}: {e}")
            return False
    
    def list_conversations(self) -> List[str]:
        """List all available conversation IDs"""
        conversation_ids = []
        
        # From disk
        for file_path in self.storage_dir.glob("*.json"):
            conversation_ids.append(file_path.stem)
        
        # From memory (might not be saved yet)
        for conv_id in self._active_conversations.keys():
            if conv_id not in conversation_ids:
                conversation_ids.append(conv_id)
        
        return sorted(conversation_ids)
    
    def cleanup_old_conversations(self, days: int = 30) -> int:
        """Clean up conversations older than specified days"""
        cutoff_time = time.time() - (days * 24 * 60 * 60)
        cleaned_count = 0
        
        for conversation_id in self.list_conversations():
            conversation = self.load_conversation(conversation_id)
            if conversation and conversation.last_updated < cutoff_time:
                if self.clear_conversation(conversation_id):
                    cleaned_count += 1
        
        logger.info(f"Cleaned up {cleaned_count} old conversations")
        return cleaned_count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get conversation statistics"""
        all_conversations = self.list_conversations()
        active_count = len(self._active_conversations)
        
        # Calculate total messages
        total_messages = 0
        for conv_id in all_conversations:
            conversation = self.load_conversation(conv_id)
            if conversation:
                total_messages += len(conversation.messages)
        
        return {
            'total_conversations': len(all_conversations),
            'active_conversations': active_count,
            'total_messages': total_messages,
            'storage_directory': str(self.storage_dir),
            'last_cleanup': self._last_cleanup
        }
    
    def periodic_cleanup(self) -> None:
        """Perform periodic cleanup of memory cache"""
        current_time = time.time()
        
        # Only cleanup if it's been more than an hour
        if current_time - self._last_cleanup < 3600:
            return
        
        # Remove old conversations from memory cache
        max_conversations = config.conversation.MAX_CONVERSATIONS_IN_MEMORY
        if len(self._active_conversations) > max_conversations:
            # Sort by last updated time and remove oldest
            sorted_conversations = sorted(
                self._active_conversations.items(),
                key=lambda x: x[1].last_updated
            )
            
            # Remove oldest conversations
            to_remove = len(self._active_conversations) - max_conversations
            for i in range(to_remove):
                conv_id, conversation = sorted_conversations[i]
                
                # Save before removing from memory
                self.save_conversation(conversation)
                del self._active_conversations[conv_id]
                logger.debug(f"Removed conversation from memory: {conv_id}")
        
        self._last_cleanup = current_time
        logger.info("Periodic cleanup completed")
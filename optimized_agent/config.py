"""
Configuration management for PartSelect Chat Agent
"""
import os
from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path

LLM_MODEL = "gemma3:latest"

@dataclass
class DatabaseConfig:
    """Database and storage configuration"""
    DOCS_FOLDER: str = "documents"
    VECTOR_DB_PATH: str = "vector_db"
    CACHE_DIR: str = "./llm_cache"
    SCRAPED_DATA_DIR: str = "scraped_data"
    CONVERSATION_HISTORY_DIR: str = "conversations"


@dataclass
class ModelConfig:
    """Model configuration for embeddings and LLM"""
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    LLM_MODEL_NAME: str = LLM_MODEL
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    DEEPSEEK_MODEL_NAME: str = "deepseek-chat"  # For future DeepSeek integration
    
    # LLM Parameters
    TEMPERATURE: float = 0.1
    TOP_P: float = 0.9
    MAX_TOKENS: int = 500
    REQUEST_TIMEOUT: int = 180


@dataclass
class ProcessingConfig:
    """Text processing and retrieval configuration"""
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    SIMILARITY_THRESHOLD: float = 0.1
    MAX_RETRIEVAL_CHUNKS: int = 5
    MAX_CONTEXT_LENGTH: int = 2000
    SEMANTIC_SIMILARITY_THRESHOLD: float = 0.85


@dataclass
class ScrapingConfig:
    """Web scraping configuration"""
    BASE_URL: str = "https://www.partselect.com"
    DEFAULT_TIMEOUT: int = 15
    MAX_RETRIES: int = 3
    RETRY_MIN_WAIT: int = 1
    RETRY_MAX_WAIT: int = 10
    PRODUCTS: List[str] = field(default_factory=lambda: ["Refrigerator", "Dishwasher"])
    
    # Selenium options
    HEADLESS: bool = True
    NO_SANDBOX: bool = True
    DISABLE_GPU: bool = True


@dataclass
class ConversationConfig:
    """Conversation management configuration"""
    MAX_CONVERSATION_HISTORY: int = 50
    CONTEXT_WINDOW_SIZE: int = 6
    AUTO_SAVE_INTERVAL: int = 10  # Save every N messages
    MAX_CONVERSATIONS_IN_MEMORY: int = 100


@dataclass
class SecurityConfig:
    """Security and privacy configuration"""
    ALLOWED_DOMAINS: List[str] = field(default_factory=lambda: ["partselect.com"])
    RATE_LIMIT_PER_HOUR: int = 1000
    MAX_MESSAGE_LENGTH: int = 1000
    ENABLE_CONTENT_FILTERING: bool = True


class SystemPrompts:
    """System prompts and templates"""
    
    BASE_SYSTEM_PROMPT = """You are a specialized PartSelect customer service assistant for appliance parts and repairs.

    CORE FUNCTIONS:
    1. get_part_information(model_number): Get detailed information about a specific part/model
    2. check_model_part_compatibility(model_number, part_number): Check if a part is compatible with a model

    FUNCTION USAGE RULES:
    - Use get_part_information() when user asks about a specific part/model number
    - Use check_model_part_compatibility() when user asks about compatibility between two items
    - Track part numbers and model numbers mentioned in conversation
    - When user says "this part" or "that part", refer to the most recently mentioned part number
    - When user says "my model", refer to the most recently mentioned model number

    RESPONSE GUIDELINES:
    - Provide clear, step-by-step repair instructions when requested
    - Always mention safety precautions for repairs
    - Include required tools and estimated difficulty/time
    - Stay focused on refrigerator and dishwasher parts only
    - End responses with helpful follow-up questions
    - Use conversation context to understand references like "this part" or "my model"

    If you need to call a function, respond with EXACTLY this format:
    [FUNCTION_NAME(parameter="value")]

    Examples:
    - [get_part_information(model_number="PS11752778")]
    - [check_model_part_compatibility(model_number="WDT780SAEM1", part_number="PS11752778")]
    """

    CONTEXT_TEMPLATE = """
    CONVERSATION CONTEXT: {context_summary}

    RECENT CONVERSATION:
    {conversation_history}

    KNOWLEDGE BASE CONTEXT:
    {knowledge_context}

    CURRENT USER QUERY: {user_query}

    Remember: When user refers to "this part", "that part", "my model", etc., use the conversation context to identify the specific items they're referring to.
    """


@dataclass
class AppConfig:
    """Main application configuration"""
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    scraping: ScrapingConfig = field(default_factory=ScrapingConfig)
    conversation: ConversationConfig = field(default_factory=ConversationConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    
    # Environment-based overrides
    def __post_init__(self):
        # Override with environment variables if present
        self.model.OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", self.model.OLLAMA_BASE_URL)
        self.model.LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", self.model.LLM_MODEL_NAME)
        
        # Create necessary directories
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directories"""
        directories = [
            self.database.DOCS_FOLDER,
            self.database.VECTOR_DB_PATH,
            self.database.CACHE_DIR,
            self.database.SCRAPED_DATA_DIR,
            self.database.CONVERSATION_HISTORY_DIR
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)


# Regex patterns for entity extraction
class EntityPatterns:
    """Regular expression patterns for extracting entities"""
    
    PART_NUMBER_PATTERNS = [
        r'\b(PS\d{8,})\b',           # PartSelect numbers
        r'\b(WP[A-Z0-9]{8,})\b',     # Whirlpool parts
        r'\b([A-Z]{2,3}\d{6,})\b',   # Generic part numbers
        r'\b(W\d{8,})\b',            # W-prefix parts
    ]
    
    MODEL_NUMBER_PATTERNS = [
        r'\b([A-Z]{3,4}\d{6,12}[A-Z]?\d{0,2})\b',  # Standard appliance models
        r'\b(WDT\w+)\b',             # Whirlpool dishwasher models
        r'\b(WRS\w+)\b',             # Whirlpool refrigerator models
        r'\b(WRF\w+)\b',             # Whirlpool French door refrigerators
        r'\b(WRB\w+)\b',             # Whirlpool bottom freezer refrigerators
    ]


# CSS Selectors for web scraping
class WebSelectors:
    """CSS selectors for web scraping"""
    
    SEARCH_INPUT = '#searchboxInput'
    SEARCH_BUTTON = 'button.js-searchBtn'
    POPUP_DECLINE = 'button.bx-button[type="reset"][aria-label="Decline; close the dialog"]'
    POPUP_SLAB = '.bx-slab'
    PART_SEARCH_INPUT = 'input[aria-label="Enter a part description"]'
    PART_SEARCH_BUTTON = 'button.search-btn[aria-label="search"]'
    
    PART_INFO = {
        'partselect_number': '[itemprop="productID"]',
        'manufacturer_part_number': '[itemprop="mpn"]',
        'title': 'h1[itemprop="name"]',
        'price': 'span.price[itemprop="price"]',
        'availability': '[itemprop="availability"]',
        'description': '[itemprop="description"]',
        'reviews': '.rating__count',
        'main_image_url': '.main-media.MagicZoom-PartImage img',
        'brand': '[itemprop="brand"] [itemprop="name"]'
    }


# Default configuration instance
config = AppConfig()
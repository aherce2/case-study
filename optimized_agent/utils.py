"""
Utility functions for the PartSelect chat agent
"""
import re
import os
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import unicodedata

logger = logging.getLogger(__name__)


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Split text into overlapping chunks with improved sentence boundary detection
    
    Args:
        text: Input text to chunk
        chunk_size: Maximum size of each chunk
        overlap: Number of characters to overlap between chunks
        
    Returns:
        List of text chunks
    """
    if not text or len(text) <= chunk_size:
        return [text] if text else []
    
    chunks = []
    text = text.strip()
    
    # Split by sentences first for better chunking
    sentences = _split_into_sentences(text)
    
    current_chunk = ""
    current_size = 0
    
    for sentence in sentences:
        sentence_size = len(sentence)
        
        # If adding this sentence would exceed chunk_size
        if current_size + sentence_size > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            
            # Start new chunk with overlap from previous chunk
            if overlap > 0 and len(current_chunk) > overlap:
                overlap_text = current_chunk[-overlap:].strip()
                current_chunk = overlap_text + " " + sentence
                current_size = len(current_chunk)
            else:
                current_chunk = sentence
                current_size = sentence_size
        else:
            # Add sentence to current chunk
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
            current_size = len(current_chunk)
    
    # Add the last chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    logger.debug(f"Split text of {len(text)} chars into {len(chunks)} chunks")
    return chunks


def _split_into_sentences(text: str) -> List[str]:
    """Split text into sentences using improved regex patterns"""
    # Improved sentence boundary detection
    sentence_endings = r'[.!?]+(?=\s+[A-Z]|\s*$)'
    sentences = re.split(sentence_endings, text)
    
    # Clean up sentences and remove empty ones
    cleaned_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence:
            cleaned_sentences.append(sentence)
    
    return cleaned_sentences


def sanitize_filename(filename: str, max_length: int = 255) -> str:
    """
    Sanitize filename by removing invalid characters and limiting length
    
    Args:
        filename: Original filename
        max_length: Maximum length for the filename
        
    Returns:
        Sanitized filename
    """
    # Normalize unicode characters
    filename = unicodedata.normalize('NFKD', filename)
    
    # Remove or replace invalid characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    filename = re.sub(r'[^\w\s\-_.]', '', filename)
    filename = re.sub(r'[\s]+', '_', filename)
    filename = filename.strip('._')
    
    # Limit length
    if len(filename) > max_length:
        name, ext = os.path.splitext(filename)
        max_name_length = max_length - len(ext)
        filename = name[:max_name_length] + ext
    
    return filename or "untitled"


def generate_hash(content: str, algorithm: str = 'md5') -> str:
    """
    Generate hash for content
    
    Args:
        content: Content to hash
        algorithm: Hashing algorithm ('md5', 'sha1', 'sha256')
        
    Returns:
        Hexadecimal hash string
    """
    if algorithm == 'md5':
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    elif algorithm == 'sha1':
        return hashlib.sha1(content.encode('utf-8')).hexdigest()
    elif algorithm == 'sha256':
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")


def ensure_directory(path: str) -> Path:
    """
    Ensure directory exists, create if it doesn't
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def clean_text(text: str) -> str:
    """
    Clean text by removing extra whitespace and normalizing
    
    Args:
        text: Input text
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    # Remove control characters but keep newlines and tabs
    text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t')
    
    return text


def extract_numbers(text: str) -> List[str]:
    """
    Extract number patterns from text (part numbers, model numbers, etc.)
    
    Args:
        text: Input text
        
    Returns:
        List of extracted numbers
    """
    # Pattern to match various number formats
    patterns = [
        r'\b[A-Z]{2,4}\d{6,12}[A-Z]?\d{0,2}\b',  # Model numbers
        r'\b(PS|WP)\d{6,}\b',                      # Part numbers
        r'\b[A-Z]\d{8,}\b',                        # Generic codes
        r'\b\d{6,}\b'                              # Numeric codes
    ]
    
    numbers = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        numbers.extend([match.upper() if isinstance(match, str) else match for match in matches])
    
    # Remove duplicates while preserving order
    return list(dict.fromkeys(numbers))


def format_response_text(text: str) -> str:
    """
    Format response text for better readability
    
    Args:
        text: Raw response text
        
    Returns:
        Formatted response text
    """
    if not text:
        return ""
    
    # Clean up extra whitespace
    text = clean_text(text)
    
    # Add proper spacing around bullet points
    text = re.sub(r'([.!?])\s*[•·-]\s*', r'\1\n\n• ', text)
    text = re.sub(r'^[•·-]\s*', '• ', text, flags=re.MULTILINE)
    
    # Ensure proper paragraph spacing
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Format numbered lists
    text = re.sub(r'^(\d+)\.\s*', r'\1. ', text, flags=re.MULTILINE)
    
    return text.strip()


def validate_part_number(part_number: str) -> bool:
    """
    Validate if a string looks like a valid part number
    
    Args:
        part_number: String to validate
        
    Returns:
        True if it looks like a valid part number
    """
    if not part_number:
        return False
    
    part_number = part_number.strip().upper()
    
    # Common part number patterns
    patterns = [
        r'^PS\d{8,}$',           # PartSelect numbers
        r'^WP[A-Z0-9]{6,}$',     # Whirlpool parts
        r'^[A-Z]{2,3}\d{6,}$',   # Generic part numbers
        r'^W\d{8,}$'             # W-prefix parts
    ]
    
    return any(re.match(pattern, part_number) for pattern in patterns)


def validate_model_number(model_number: str) -> bool:
    """
    Validate if a string looks like a valid model number
    
    Args:
        model_number: String to validate
        
    Returns:
        True if it looks like a valid model number
    """
    if not model_number:
        return False
    
    model_number = model_number.strip().upper()
    
    # Common model number patterns
    patterns = [
        r'^[A-Z]{3,4}\d{6,12}[A-Z]?\d{0,2}$',  # Standard appliance models
        r'^WDT\w+$',                            # Whirlpool dishwashers
        r'^WRS\w+$',                            # Whirlpool refrigerators
        r'^WRF\w+$',                            # Whirlpool French door refrigerators
        r'^WRB\w+$'                             # Whirlpool bottom freezer refrigerators
    ]
    
    return any(re.match(pattern, model_number) for pattern in patterns)


def parse_price(price_text: str) -> Optional[float]:
    """
    Parse price from text
    
    Args:
        price_text: Text containing price
        
    Returns:
        Parsed price as float or None
    """
    if not price_text:
        return None
    
    # Extract numeric price
    price_match = re.search(r'\$?(\d+(?:\.\d{2})?)', price_text.replace(',', ''))
    
    if price_match:
        try:
            return float(price_match.group(1))
        except ValueError:
            pass
    
    return None


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """
    Truncate text to maximum length with suffix
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add when truncated
        
    Returns:
        Truncated text
    """
    if not text or len(text) <= max_length:
        return text
    
    # Try to truncate at word boundary
    truncated = text[:max_length - len(suffix)]
    last_space = truncated.rfind(' ')
    
    if last_space > max_length * 0.7:  # If we find a space in the last 30%
        truncated = truncated[:last_space]
    
    return truncated + suffix


def measure_text_similarity(text1: str, text2: str) -> float:
    """
    Simple text similarity measure using character overlap
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        Similarity score between 0 and 1
    """
    if not text1 or not text2:
        return 0.0
    
    # Convert to lowercase and get character sets
    set1 = set(text1.lower())
    set2 = set(text2.lower())
    
    # Calculate Jaccard similarity
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    return intersection / union if union > 0 else 0.0


def get_file_info(file_path: str) -> Dict[str, Any]:
    """
    Get comprehensive file information
    
    Args:
        file_path: Path to file
        
    Returns:
        Dictionary with file information
    """
    path = Path(file_path)
    
    if not path.exists():
        return {"exists": False}
    
    stat = path.stat()
    
    return {
        "exists": True,
        "name": path.name,
        "stem": path.stem,
        "suffix": path.suffix,
        "size": stat.st_size,
        "modified_time": stat.st_mtime,
        "created_time": stat.st_ctime,
        "is_file": path.is_file(),
        "is_directory": path.is_dir(),
        "absolute_path": str(path.absolute())
    }


def batch_process(items: List[Any], batch_size: int = 100) -> List[List[Any]]:
    """
    Split items into batches for processing
    
    Args:
        items: List of items to batch
        batch_size: Size of each batch
        
    Returns:
        List of batches
    """
    batches = []
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        batches.append(batch)
    
    return batches
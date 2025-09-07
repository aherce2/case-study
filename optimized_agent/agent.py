
"""
Enhanced PartSelect RAG Agent with improved architecture and scalability
"""
import os
import json
import pickle
import hashlib
import logging
import time
import requests
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple, Any
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Import our modular components
from config import config, SystemPrompts
from conversation_manager import ConversationManager, ConversationContext
from handler import FunctionManager, FunctionCallParser
from optimized_scraping import PartSelectScraper
from semantic_cache import OllamaSemanticCache
from utils import chunk_text, sanitize_filename

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """Structured document chunk for better type safety"""
    id: str
    text: str
    filename: str
    source_url: str = ""
    chunk_id: int = 0
    similarity_score: float = 0.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class RetrievalResult:
    """Result from document retrieval"""
    chunks: List[DocumentChunk]
    query: str
    total_found: int
    processing_time: float = 0.0


class VectorDatabase:
    """Vector database operations with improved error handling"""
    
    def __init__(self, db_path: str, embedding_model: SentenceTransformer):
        self.db_path = Path(db_path)
        self.embedding_model = embedding_model
        
        # File paths
        self.docs_file = self.db_path / "documents.json"
        self.chunks_file = self.db_path / "chunks.json"
        self.embeddings_file = self.db_path / "embeddings.pkl"
        self.metadata_file = self.db_path / "metadata.json"
        
        # Data storage
        self.documents: List[Dict] = []
        self.chunks: List[DocumentChunk] = []
        self.embeddings: Optional[np.ndarray] = None
        
        # Create directory
        self.db_path.mkdir(parents=True, exist_ok=True)
    
    def save(self) -> bool:
        """Save vector database to disk"""
        try:
            with open(self.docs_file, "w", encoding="utf-8") as f:
                json.dump(self.documents, f, indent=2)
            
            chunks_dict = [asdict(chunk) for chunk in self.chunks]
            with open(self.chunks_file, "w", encoding="utf-8") as f:
                json.dump(chunks_dict, f, indent=2)
            
            if self.embeddings is not None:
                with open(self.embeddings_file, "wb") as f:
                    pickle.dump(self.embeddings, f)
            
            metadata = {
                "num_documents": len(self.documents),
                "num_chunks": len(self.chunks),
                "embeddings_shape": self.embeddings.shape if self.embeddings is not None else None,
                "last_updated": str(Path().resolve()),
                "config_version": "2.0"
            }
            with open(self.metadata_file, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
                
            logger.info(f"Vector database saved to {self.db_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving vector database: {e}")
            return False

    def load(self) -> bool:
        """Load vector database from disk"""
        try:
            if not all(f.exists() for f in [self.docs_file, self.chunks_file]):
                return False
            
            with open(self.docs_file, "r", encoding="utf-8") as f:
                self.documents = json.load(f)
            
            with open(self.chunks_file, "r", encoding="utf-8") as f:
                chunks_data = json.load(f)
                self.chunks = [DocumentChunk(**chunk_data) for chunk_data in chunks_data]
            
            if self.embeddings_file.exists():
                with open(self.embeddings_file, "rb") as f:
                    self.embeddings = pickle.load(f)
            
            logger.info(f"Loaded vector database: {len(self.documents)} docs, {len(self.chunks)} chunks")
            return True
        except Exception as e:
            logger.error(f"Error loading vector database: {e}")
            return False

    def is_valid(self) -> bool:
        """Check if database is valid and loaded"""
        return (
            len(self.documents) > 0 and 
            len(self.chunks) > 0 and 
            self.embeddings is not None
        )


class DocumentProcessor:
    """Document processor with improved chunking"""
    
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or config.processing.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or config.processing.CHUNK_OVERLAP
        
    def load_documents(self, docs_folder: str) -> List[Dict]:
        """Load documents from folder with enhanced metadata"""
        documents = []
        docs_path = Path(docs_folder)
        
        if not docs_path.exists():
            logger.warning(f"Documents folder not found: {docs_folder}")
            return documents
        
        for file_path in docs_path.rglob("*.txt"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                
                if content:
                    doc_id = hashlib.md5(str(file_path).encode()).hexdigest()
                    documents.append({
                        "id": doc_id,
                        "filename": file_path.name,
                        "content": content,
                        "path": str(file_path),
                        "size": len(content),
                        "modified_time": file_path.stat().st_mtime
                    })
                    
            except Exception as e:
                logger.error(f"Error loading document {file_path}: {e}")
        
        logger.info(f"Loaded {len(documents)} documents")
        return documents
    
    def create_chunks(self, documents: List[Dict]) -> List[DocumentChunk]:
        """Create chunks from documents with improved metadata"""
        chunks = []
        
        for doc in documents:
            doc_chunks = chunk_text(
                doc["content"], 
                chunk_size=self.chunk_size,
                overlap=self.chunk_overlap
            )
            
            for i, chunk_text in enumerate(doc_chunks):
                chunk_id = hashlib.md5(f"{doc['id']}_{i}_{chunk_text[:50]}".encode()).hexdigest()
                
                chunk = DocumentChunk(
                    id=chunk_id,
                    text=chunk_text,
                    filename=doc["filename"],
                    chunk_id=i,
                    metadata={
                        "document_id": doc["id"],
                        "document_path": doc["path"],
                        "chunk_index": i,
                        "total_chunks": len(doc_chunks),
                        "document_size": doc["size"]
                    }
                )
                chunks.append(chunk)
        
        logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
        return chunks


class LLMService:
    """LLM service with improved error handling and caching"""
    
    def __init__(self, model_name: str = None, base_url: str = None):
        self.model_name = model_name or config.model.LLM_MODEL_NAME
        self.base_url = base_url or config.model.OLLAMA_BASE_URL
        self.semantic_cache = OllamaSemanticCache()
        
    def generate_response(self, prompt: str, conversation_context: ConversationContext) -> str:
        """Generate response with conversation context and caching"""
        try:
            # Build enhanced prompt with conversation context
            enhanced_prompt = self._build_enhanced_prompt(prompt, conversation_context)
            
            # Check semantic cache first
            # cached_response = self.semantic_cache.get_response(enhanced_prompt, self.model_name)
            # if cached_response:
            #     return cached_response.get("response", {}).get("response", "")
            cached_response = self.semantic_cache.get_response(enhanced_prompt, self.model_name)
            if cached_response:
                if isinstance(cached_response, dict):
                    return cached_response.get("response", {}).get("response", "")
                elif isinstance(cached_response, str):
                    return cached_response

            # Make direct API call
            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": enhanced_prompt}],
                "stream": False,
                "options": {
                    "temperature": config.model.TEMPERATURE,
                    "top_p": config.model.TOP_P,
                    "max_tokens": config.model.MAX_TOKENS,
                }
            }
            
            response = requests.post(
                f"{self.base_url}/api/chat", 
                json=payload, 
                timeout=config.model.REQUEST_TIMEOUT
            )
            
            if response.status_code == 200:
                result = response.json()["message"]["content"].strip()
                return result
            else:
                return f"Error: LLM server returned status {response.status_code}"
                
        except requests.exceptions.ConnectionError:
            return "Error: Cannot connect to LLM server. Please ensure Ollama is running."
        except requests.exceptions.Timeout:
            return "Error: LLM request timed out. Please try again."
        except Exception as e:
            logger.error(f"Error in LLM generation: {e}")
            return f"Error: {str(e)}"
    
    def _build_enhanced_prompt(self, prompt: str, context: ConversationContext) -> str:
        """Build enhanced prompt with conversation context"""
        return SystemPrompts.CONTEXT_TEMPLATE.format(
            context_summary=context.get_context_summary(),
            conversation_history=context.format_conversation_history(),
            knowledge_context=prompt,
            user_query=context.messages[-1].content if context.messages else ""
        )


class PartSelectAgent:
    """Main PartSelect agent with improved architecture"""
    
    def __init__(self, custom_config: Dict[str, Any] = None):
        # Apply custom configuration if provided
        if custom_config:
            self._apply_custom_config(custom_config)
        
        # Initialize components
        logger.info("Initializing PartSelect Agent...")
        self._init_components()
        logger.info("Agent initialization complete!")
    
    def _apply_custom_config(self, custom_config: Dict[str, Any]) -> None:
        """Apply custom configuration settings"""
        for section, values in custom_config.items():
            if hasattr(config, section):
                section_config = getattr(config, section)
                for key, value in values.items():
                    if hasattr(section_config, key):
                        setattr(section_config, key, value)
                        logger.info(f"Applied config: {section}.{key} = {value}")
    
    def _init_components(self) -> None:
        """Initialize all agent components"""
        # Core models
        self.embedding_model = SentenceTransformer(config.model.EMBEDDING_MODEL_NAME)
        
        # Services
        self.llm_service = LLMService()
        self.conversation_manager = ConversationManager()
        
        # Scraping and functions
        self.scraper = PartSelectScraper(
            products=config.scraping.PRODUCTS,
            timeout=config.scraping.DEFAULT_TIMEOUT,
            save_dir=config.database.SCRAPED_DATA_DIR
        )
        self.function_manager = FunctionManager(self.scraper)
        
        # Vector database and document processing
        self.vector_db = VectorDatabase(config.database.VECTOR_DB_PATH, self.embedding_model)
        self.doc_processor = DocumentProcessor()
        
        # Simple query cache for exact matches
        self.query_cache: Dict[str, str] = {}
    
    def build_index(self, force_rebuild: bool = False) -> bool:
        """Build or load the vector index"""
        if not force_rebuild and self.vector_db.load() and self.vector_db.is_valid():
            logger.info("Vector index loaded from cache")
            return True
        
        logger.info("Building vector index...")
        documents = self.doc_processor.load_documents(config.database.DOCS_FOLDER)
        if not documents:
            logger.warning("No documents found!")
            return False
        
        chunks = self.doc_processor.create_chunks(documents)
        
        logger.info("Creating embeddings...")
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        
        self.vector_db.documents = documents
        self.vector_db.chunks = chunks
        self.vector_db.embeddings = embeddings
        
        return self.vector_db.save()
    
    def retrieve_relevant_chunks(self, query: str, top_k: Optional[int] = None) -> RetrievalResult:
        """Retrieve relevant document chunks for a query"""
        start_time = time.time()
        top_k = top_k or config.processing.MAX_RETRIEVAL_CHUNKS
        
        if not self.vector_db.is_valid():
            self.build_index()
        
        if not self.vector_db.is_valid():
            return RetrievalResult(chunks=[], query=query, total_found=0)
        
        query_embedding = self.embedding_model.encode([query])
        similarities = cosine_similarity(query_embedding, self.vector_db.embeddings)[0]
        top_indices = similarities.argsort()[::-1][:top_k]
        
        relevant_chunks = []
        for idx in top_indices:
            if similarities[idx] > config.processing.SIMILARITY_THRESHOLD:
                chunk = self.vector_db.chunks[idx]
                relevant_chunk = DocumentChunk(
                    id=chunk.id,
                    text=chunk.text,
                    filename=chunk.filename,
                    source_url=chunk.source_url,
                    chunk_id=chunk.chunk_id,
                    similarity_score=float(similarities[idx]),
                    metadata=chunk.metadata
                )
                relevant_chunks.append(relevant_chunk)
        
        processing_time = time.time() - start_time
        return RetrievalResult(
            chunks=relevant_chunks,
            query=query,
            total_found=len(relevant_chunks),
            processing_time=processing_time
        )
    
    def _build_knowledge_context(self, chunks: List[DocumentChunk]) -> str:
        """Build context string from relevant chunks"""
        if not chunks:
            return "No relevant documentation found."
        
        context_parts = []
        seen_texts = set()
        total_length = 0
        
        for chunk in chunks:
            snippet = chunk.text.strip()
            if snippet not in seen_texts and total_length < config.processing.MAX_CONTEXT_LENGTH:
                remaining_space = config.processing.MAX_CONTEXT_LENGTH - total_length
                if len(snippet) > remaining_space:
                    snippet = snippet[:remaining_space-3] + "..."
                
                context_parts.append(f"[Relevance: {chunk.similarity_score:.2f}] {snippet}")
                seen_texts.add(snippet)
                total_length += len(snippet)
        
        return "\n\n".join(context_parts)
    
    def chat(self, user_input: str, conversation_id: str = "default") -> str:
        """Enhanced chat interface with improved function calling"""
        # Get conversation context
        conversation = self.conversation_manager.get_or_create_conversation(conversation_id)
        
        # Resolve references using conversation context
        resolved_input = conversation.resolve_references(user_input)
        
        # Add user message to conversation
        conversation = self.conversation_manager.add_message(
            conversation_id, "user", user_input, auto_save=True
        )
        
        # Check exact query cache first
        cache_key = f"{conversation_id}::{resolved_input.lower().strip()}"
        if cache_key in self.query_cache:
            logger.info("Exact query cache hit")
            cached_response = self.query_cache[cache_key]
            self.conversation_manager.add_message(
                conversation_id, "assistant", cached_response, auto_save=True
            )
            return cached_response
        
        # Retrieve relevant documentation
        retrieval_result = self.retrieve_relevant_chunks(resolved_input)
        knowledge_context = self._build_knowledge_context(retrieval_result.chunks)
        
        # Build full prompt
        full_prompt = f"""
        CURRENT USER QUERY: {resolved_input}

        RELEVANT DOCUMENTATION:
        {knowledge_context}

        Please respond appropriately. If you need to call a function, use the exact format specified in the system prompt.
        Please provide detailed information about the part, including links, specifications, and descriptions. 
        Include all relevant data from the knowledge base and avoid only asking follow-up questions unless necessary.

        """
                
        # Generate LLM response
        response = self.llm_service.generate_response(full_prompt, conversation)
        
        # Process function calls
        processed_response, function_results = self.function_manager.process_response(response)
        
        # Add assistant response to conversation
        response_metadata = {
            "retrieval_chunks": len(retrieval_result.chunks),
            "function_calls": len(function_results),
            "processing_time": retrieval_result.processing_time
        }
        
        self.conversation_manager.add_message(
            conversation_id, "assistant", processed_response, 
            metadata=response_metadata, auto_save=True
        )
        
        # Cache the response
        self.query_cache[cache_key] = processed_response
        
        # Periodic cleanup
        self.conversation_manager.periodic_cleanup()
        
        return processed_response
    
    def get_conversation_history(self, conversation_id: str, limit: int = None) -> List[Dict[str, Any]]:
        """Get conversation history"""
        return self.conversation_manager.get_conversation_history(conversation_id, limit)
    
    def clear_conversation(self, conversation_id: str) -> bool:
        """Clear conversation"""
        return self.conversation_manager.clear_conversation(conversation_id)
    
    def list_conversations(self) -> List[str]:
        """List all conversations"""
        return self.conversation_manager.list_conversations()
    
    def get_agent_stats(self) -> Dict[str, Any]:
        """Get comprehensive agent statistics"""
        return {
            "config": {
                "model_name": config.model.LLM_MODEL_NAME,
                "embedding_model": config.model.EMBEDDING_MODEL_NAME,
                "chunk_size": config.processing.CHUNK_SIZE,
                "similarity_threshold": config.processing.SIMILARITY_THRESHOLD
            },
            "vector_db": {
                "documents": len(self.vector_db.documents),
                "chunks": len(self.vector_db.chunks),
                "embeddings_shape": self.vector_db.embeddings.shape if self.vector_db.embeddings is not None else None,
            },
            "conversations": self.conversation_manager.get_stats(),
            "functions": list(self.function_manager.get_available_functions().keys()),
            "cache_size": len(self.query_cache)
        }


# semantic_cache.py
# Source Code: https://oleg-dubetcky.medium.com/mastering-caching-llm-calls-with-tracing-and-retrying-63e12c3318ef
import os
import time
import logging
import hashlib
import numpy as np
from config import LLM_MODEL
from diskcache import Cache
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)

from sklearn.metrics.pairwise import cosine_similarity
from ollama import Client

# ---------------- CONFIGURATION ----------------
class Config:
    CACHE_DIR = "./llm_cache"
    EMBED_MODEL = "mxbai-embed-large"
    GEN_MODEL = LLM_MODEL
    SIMILARITY_THRESHOLD = 0.85
    MAX_RETRIES = 3
    RETRY_MIN_WAIT = 1
    RETRY_MAX_WAIT = 10
    OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# ---------------- LOGGING ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------- RETRY CONFIGURATION ----------------
def before_sleep_callback(retry_state):
    logger.warning(
        f"Retrying {retry_state.fn.__name__} after failure "
        f"(attempt {retry_state.attempt_number}): {str(retry_state.outcome.exception())}"
    )

retry_config = {
    "stop": stop_after_attempt(Config.MAX_RETRIES),
    "wait": wait_exponential(
        multiplier=1,
        min=Config.RETRY_MIN_WAIT,
        max=Config.RETRY_MAX_WAIT
    ),
    "retry": retry_if_exception_type((Exception,)), 
    "before_sleep": before_sleep_callback,
    "reraise": True
}

# ---------------- CACHE CLASS ----------------
class OllamaSemanticCache:
    def __init__(self, cache_dir=Config.CACHE_DIR, embedding_model=Config.EMBED_MODEL):
        self.cache = Cache(cache_dir)
        self.embedding_model_name = embedding_model
        self.client = Client(host=Config.OLLAMA_HOST)

    @retry(**retry_config)
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding vector with retries"""
        response = self.client.embeddings(model=self.embedding_model_name, prompt=text)
        return np.array(response["embedding"])

    def find_similar(self, embedding: np.ndarray, threshold=Config.SIMILARITY_THRESHOLD):
        """Search cache for semantically similar cached embeddings"""
        for key in self.cache:
            # Filter keys to embedding keys only (tuples)
            if not isinstance(key, tuple):
                continue
            try:
                cached_embed = np.array(key)
                sim = cosine_similarity([embedding], [cached_embed])[0][0]
                if sim > threshold:
                    logger.info(f"Cache hit with similarity {sim:.2f}")
                    return self.cache[key]
            except Exception as e:
                logger.warning(f"Cache similarity check failed: {str(e)}")
        return None

    @retry(**retry_config)
    def generate_response(self, prompt: str, model: str) -> dict:
        """Call Ollama to generate a response with retries"""
        start = time.time()
        response = self.client.generate(model=model, prompt=prompt)
        latency = time.time() - start
        return {"response": response, "latency": latency}

    def get_response(self, prompt: str, model: str = Config.GEN_MODEL) -> dict:
        """Main interface that uses semantic caching and retries"""
        logger.info(f"Processing prompt: {prompt[:60]}...")
        
        # Get embedding (with retry)
        embedding = self.get_embedding(prompt)

        # Check semantic cache
        cached = self.find_similar(embedding)
        if cached:
            return cached

        # Cache miss; generate new response
        result = self.generate_response(prompt, model)
        
        # Store response indexed by embedding vector and prompt hash
        self.cache[tuple(embedding)] = result["response"]
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
        self.cache[prompt_hash] = result["response"]

        return result

    def close(self):
        """Properly close the cache"""
        self.cache.close()


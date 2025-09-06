import os
import json
import pickle
import numpy as np
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import requests
from datetime import datetime
import re

products = ["refrigerator", "dishwasher"]
product_list_str = ", ".join(products)

BASE_PROMPT = f"""
You are a specialized PartSelect customer service agent. You help customers with {product_list_str} parts, repairs, and troubleshooting.

IMPORTANT GUIDELINES:
- Only answer questions about these products: {product_list_str}
- Use the provided context from PartSelect documentation to give accurate answers
- If you don't have specific information, say so and suggest alternatives
- Provide step-by-step instructions when appropriate
- Always be helpful and professional
- Focus on part compatibility, installation, and troubleshooting

If asked about other appliances or topics outside {product_list_str} repairs, politely redirect the conversation.
"""

class OllamaRAGAgent:
    """
    Complete RAG Agent that loads txt files, creates embeddings, and uses Ollama DeepSeek for responses
    """
    
    def __init__(self, 
                docs_folder: str,
                vector_db_path: str,
                ollama_base_url: str,
                embedding_model: str,
                llm_model: str ,
                system_prompt: str, 
                products: Optional[List[str]]):
        """
        Initialize the RAG Agent
        
        Args:
            docs_folder: Path to folder containing txt files
            vector_db_path: Path to store vector database
            ollama_base_url: Ollama server URL
            embedding_model: Sentence transformer model for embeddings
            llm_model: Ollama model name for chat
        """
        self.docs_folder = docs_folder
        self.vector_db_path = vector_db_path
        self.ollama_base_url = ollama_base_url
        self.embedding_model_name = embedding_model
        self.llm_model = llm_model
        
        # Initialize embedding model
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Storage for documents and embeddings
        self.documents = []
        self.embeddings = None
        self.chunks = []  # Store text chunks with metadata
        
        # Create vector DB directory
        os.makedirs(vector_db_path, exist_ok=True)
        
        # File paths for persistence
        self.docs_file = os.path.join(vector_db_path, "documents.json")
        self.chunks_file = os.path.join(vector_db_path, "chunks.json")
        self.embeddings_file = os.path.join(vector_db_path, "embeddings.pkl")
        
        # Agent system prompt for dishwasher issue classification
        self.products = products if products else []
        self.system_prompt = system_prompt if system_prompt else ""

            

    
    import os

    def load_documents(self) -> List[Dict]:
        docs = []
        if not os.path.exists(self.docs_folder):
            print(f"Warning: Documents folder '{self.docs_folder}' not found!")
            return docs

        print(f"Loading documents recursively from {self.docs_folder} ...")

        for root, dirs, files in os.walk(self.docs_folder):
            for filename in files:
                if filename.endswith(".txt"):
                    file_path = os.path.join(root, filename)
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()

                        # Extract source URL if present at the top of the file
                        source_url = ""
                        if content.startswith("Source URL:"):
                            lines = content.split("\n", 2)
                            if len(lines) > 1:
                                source_url = lines[0].replace("Source URL:", "").strip()
                                content = lines[2] if len(lines) > 2 else lines[1]

                        relative_path = os.path.relpath(file_path, self.docs_folder)
                        product_category = relative_path.split(os.sep)[:2]  

                        docs.append({
                            "filename": filename,
                            "text": content,
                            "source_url": source_url,
                            "file_path": file_path,
                            "relative_path": relative_path,
                            "product": product_category[0] if len(product_category) > 0 else None,
                            "category": product_category[1] if len(product_category) > 1 else None
                        })

                        print(f"Loaded: {relative_path} ({len(content)} characters)")
                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")

        print(f"Loaded {len(docs)} documents in total")

        self.documents = docs
        return docs

    
    def chunk_documents(self, chunk_size: int = 500, overlap: int = 50) -> List[Dict]:
        """
        Split documents into smaller chunks for better retrieval
        
        Args:
            chunk_size: Maximum characters per chunk
            overlap: Overlap between chunks
            
        Returns:
            List of text chunks with metadata
        """
        chunks = []
        
        for doc in self.documents:
            text = doc["text"]
            filename = doc["filename"]
            source_url = doc.get("source_url", "")
            
            # Split text into sentences for better chunking
            sentences = re.split(r'[.!?]\s+', text)
            
            current_chunk = ""
            chunk_id = 0
            
            for sentence in sentences:
                # Check if adding this sentence would exceed chunk size
                if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                    # Save current chunk
                    chunks.append({
                        "id": f"{filename}_{chunk_id}",
                        "text": current_chunk.strip(),
                        "filename": filename,
                        "source_url": source_url,
                        "chunk_id": chunk_id
                    })
                    
                    # Start new chunk with overlap
                    overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                    current_chunk = overlap_text + " " + sentence
                    chunk_id += 1
                else:
                    current_chunk += " " + sentence if current_chunk else sentence
            
            # Add final chunk if it has content
            if current_chunk.strip():
                chunks.append({
                    "id": f"{filename}_{chunk_id}",
                    "text": current_chunk.strip(),
                    "filename": filename,
                    "source_url": source_url,
                    "chunk_id": chunk_id
                })
        
        print(f"Created {len(chunks)} text chunks")
        self.chunks = chunks
        return chunks
    
    def create_embeddings(self) -> np.ndarray:
        """
        Create embeddings for all text chunks
        
        Returns:
            Numpy array of embeddings
        """
        if not self.chunks:
            print("No chunks available. Please load and chunk documents first.")
            return None
        
        print("Creating embeddings for text chunks...")
        
        # Extract text from chunks
        texts = [chunk["text"] for chunk in self.chunks]
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        
        print(f"Created embeddings: {embeddings.shape}")
        self.embeddings = embeddings
        return embeddings
    
    def save_vector_db(self):
        """Save vector database to disk"""
        try:
            # Save documents
            with open(self.docs_file, 'w', encoding='utf-8') as f:
                json.dump(self.documents, f, indent=2)
            
            # Save chunks
            with open(self.chunks_file, 'w', encoding='utf-8') as f:
                json.dump(self.chunks, f, indent=2)
            
            # Save embeddings
            if self.embeddings is not None:
                with open(self.embeddings_file, 'wb') as f:
                    pickle.dump(self.embeddings, f)
            
            print(f"Vector database saved to {self.vector_db_path}")
        except Exception as e:
            print(f"Error saving vector database: {e}")
    
    def load_vector_db(self) -> bool:
        """
        Load vector database from disk
        
        Returns:
            True if successfully loaded, False otherwise
        """
        try:
            # Load documents
            if os.path.exists(self.docs_file):
                with open(self.docs_file, 'r', encoding='utf-8') as f:
                    self.documents = json.load(f)
            
            # Load chunks
            if os.path.exists(self.chunks_file):
                with open(self.chunks_file, 'r', encoding='utf-8') as f:
                    self.chunks = json.load(f)
            
            # Load embeddings
            if os.path.exists(self.embeddings_file):
                with open(self.embeddings_file, 'rb') as f:
                    self.embeddings = pickle.load(f)
            
            if self.documents and self.chunks and self.embeddings is not None:
                print(f"Loaded vector database: {len(self.documents)} docs, {len(self.chunks)} chunks, {self.embeddings.shape} embeddings")
                return True
            
        except Exception as e:
            print(f"Error loading vector database: {e}")
        
        return False
    
    def build_vector_index(self, force_rebuild: bool = False):
        """
        Build or load the vector index
        
        Args:
            force_rebuild: Force rebuilding even if saved data exists
        """
        # Try to load existing vector DB first
        if not force_rebuild and self.load_vector_db():
            return
        
        print("Building vector index...")
        
        # Load documents
        self.load_documents()
        
        if not self.documents:
            print("No documents found. Please check the documents folder.")
            return
        
        # Chunk documents
        self.chunk_documents()
        
        # Create embeddings
        self.create_embeddings()
        
        # Save to disk
        self.save_vector_db()
        
        print("Vector index built successfully!")
    
    def retrieve_relevant_chunks(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Retrieve most relevant chunks for a query
        
        Args:
            query: User query
            top_k: Number of top chunks to return
            
        Returns:
            List of relevant chunks with similarity scores
        """
        if self.embeddings is None or not self.chunks:
            print("Vector database not loaded. Building index...")
            self.build_vector_index()
        
        # Create query embedding
        query_embedding = self.embedding_model.encode([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Get top k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Return relevant chunks with scores
        relevant_chunks = []
        for idx in top_indices:
            if similarities[idx] > 0.1:  # Minimum similarity threshold
                chunk = self.chunks[idx].copy()
                chunk["similarity_score"] = float(similarities[idx])
                relevant_chunks.append(chunk)
        
        return relevant_chunks
    
    def call_ollama(self, prompt: str) -> str:
        """
        Call Ollama DeepSeek model with better error handling
        
        Args:
            prompt: Prompt to send to the model
            
        Returns:
            Model response
        """
        try:
            payload = {
                "model": self.llm_model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {
                    "temperature": 0.1,  # Lower temperature for more consistent classification
                    "top_p": 0.9,
                    "max_tokens": 200  # Short responses for classification
                }
            }
            
            response = requests.post(
                f"{self.ollama_base_url}/api/chat",
                json=payload,
                timeout=120 
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["message"]["content"].strip()
            else:
                return f"Error: Ollama server returned status {response.status_code}"
                
        except requests.exceptions.ConnectionError:
            return "Error: Cannot connect to Ollama server. Please ensure Ollama is running on localhost:11434"
        except requests.exceptions.Timeout:
            return "Error: Ollama request timed out. Please try again."
        except Exception as e:
            return f"Error: {str(e)}"
    def chat(self, user_query: str) -> str:
        # print(f"Processing query: {user_query}")

        relevant_chunks = self.retrieve_relevant_chunks(user_query, top_k=3)
        context_parts = []
        seen_texts = set()
        for chunk in relevant_chunks:
            snippet = chunk["text"].strip()
            if snippet not in seen_texts: 
                context_parts.append(snippet[:300]) 
                seen_texts.add(snippet)

        context_parts = [chunk['text'][:300] for chunk in relevant_chunks]
        context = "\n".join(context_parts) if context_parts else ""

        prompt = f"""{self.system_prompt}
        Context from documentation:
        {context}

        User question: {user_query}
        Instructions for response:
        - If the user asks for a repair, give a clear, step-by-step set of instructions in plain language.
        - Mention the estimated difficulty and time if available.
        - Only include links if they are directly relevant (no duplicates).
        - At the end, ask a helpful follow-up question (e.g., whether they want part numbers, tools needed, or video guides).
        - Be concise but friendly, like a customer service technician.

        Answer:"""
        # Query Ollama model to generate answer
        response = self.call_ollama(prompt)
        return response

    def get_stats(self) -> Dict:
        """Get statistics about the knowledge base"""
        return {
            "documents": len(self.documents),
            "chunks": len(self.chunks),
            "embeddings_shape": self.embeddings.shape if self.embeddings is not None else None,
            "files": [doc["filename"] for doc in self.documents]
        }


def main():
    """Main function to run the RAG agent"""
    # Initialize the agent
    print("Initializing Agent...")
    agent = OllamaRAGAgent(
        products=products,
        system_prompt=BASE_PROMPT,
        docs_folder="Scrapped Pages",
        vector_db_path="vector_db",
        ollama_base_url="http://localhost:11434",
        embedding_model="all-MiniLM-L6-v2",
        llm_model="gemma3:latest"
    )
    
    # Build vector index (will load from cache if available)
    agent.build_vector_index()
    
    # Show statistics
    stats = agent.get_stats()
    print(f"\nKnowledge Base Stats:")
    print(f"   Documents: {stats['documents']}")
    print(f"   Text Chunks: {stats['chunks']}")

    print("Part Select Customer Service Agent ready!")
    print("Type 'quit' or 'exit' to end.\n")
    
    # Chat loop
    while True:
        try:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ["exit", "quit", "bye"]:
                print("Thank you for contacting our service team!")
                break
            
            if not user_input:
                continue
            
            # Get response from agent
            response = agent.chat(user_input)
            print(f"Agent: {response}\n")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    main()
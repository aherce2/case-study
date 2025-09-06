# agent.py
import os
import json
import pickle
import numpy as np
import re
from typing import List, Dict, Optional
import requests
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from selenium.common.exceptions import WebDriverException
from scrape import PartSelectScraper
from utils import chunk_text, sanitize_filename

class OllamaRAGAgent:
    def __init__(
        self,
        docs_folder: str,
        vector_db_path: str,
        ollama_base_url: str,
        embedding_model_name: str,
        llm_model_name: str,
        system_prompt: str,
        products: Optional[List[str]] = None,
    ):
        self.docs_folder = docs_folder
        self.vector_db_path = vector_db_path
        self.ollama_base_url = ollama_base_url
        self.embedding_model_name = embedding_model_name
        self.llm_model_name = llm_model_name
        self.system_prompt = system_prompt
        self.products = products or []

        self.scraper = PartSelectScraper(products=self.products)
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer(embedding_model_name)

        self.documents = []
        self.chunks = []
        self.embeddings = None

        os.makedirs(vector_db_path, exist_ok=True)
        self.docs_file = os.path.join(vector_db_path, "documents.json")
        self.chunks_file = os.path.join(vector_db_path, "chunks.json")
        self.embeddings_file = os.path.join(vector_db_path, "embeddings.pkl")

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
        self.documents = docs
        print(f"Loaded {len(docs)} documents in total")
        return docs

    def chunk_documents(self, chunk_size=500, overlap=50) -> List[Dict]:
        chunks = []
        for doc in self.documents:
            text = doc["text"]
            filename = doc["filename"]
            source_url = doc.get("source_url", "")

            text_chunks = chunk_text(text, chunk_size, overlap)
            for idx, chunk in enumerate(text_chunks):
                chunks.append({
                    "id": f"{filename}_{idx}",
                    "text": chunk,
                    "filename": filename,
                    "source_url": source_url,
                    "chunk_id": idx
                })
        print(f"Created {len(chunks)} text chunks")
        self.chunks = chunks
        return chunks

    def create_embeddings(self):
        if not self.chunks:
            print("No chunks available. Please load and chunk documents first.")
            return None
        print("Creating embeddings for text chunks...")
        texts = [chunk["text"] for chunk in self.chunks]
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        self.embeddings = embeddings
        print(f"Created embeddings: {embeddings.shape}")
        return embeddings

    def save_vector_db(self):
        try:
            with open(self.docs_file, "w", encoding="utf-8") as f:
                json.dump(self.documents, f, indent=2)
            with open(self.chunks_file, "w", encoding="utf-8") as f:
                json.dump(self.chunks, f, indent=2)
            if self.embeddings is not None:
                with open(self.embeddings_file, "wb") as f:
                    pickle.dump(self.embeddings, f)
            print(f"Vector database saved to {self.vector_db_path}")
        except Exception as e:
            print(f"Error saving vector database: {e}")

    def load_vector_db(self) -> bool:
        try:
            if os.path.exists(self.docs_file):
                with open(self.docs_file, "r", encoding="utf-8") as f:
                    self.documents = json.load(f)
            if os.path.exists(self.chunks_file):
                with open(self.chunks_file, "r", encoding="utf-8") as f:
                    self.chunks = json.load(f)
            if os.path.exists(self.embeddings_file):
                with open(self.embeddings_file, "rb") as f:
                    self.embeddings = pickle.load(f)

            if self.documents and self.chunks and self.embeddings is not None:
                print(f"Loaded vector database: {len(self.documents)} docs, {len(self.chunks)} chunks, {self.embeddings.shape} embeddings")
                return True
            return False
        except Exception as e:
            print(f"Error loading vector database: {e}")
            return False

    def build_vector_index(self, force_rebuild=False):
        if not force_rebuild and self.load_vector_db():
            return
        print("Building vector index...")
        self.load_documents()
        if not self.documents:
            print("No documents found. Please check the documents folder.")
            return
        self.chunk_documents()
        self.create_embeddings()
        self.save_vector_db()
        print("Vector index built successfully!")

    def retrieve_relevant_chunks(self, query: str, top_k=5) -> List[Dict]:
        if self.embeddings is None or not self.chunks:
            print("Vector database not loaded. Building index...")
            self.build_vector_index()

        query_embedding = self.embedding_model.encode([query])
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        top_indices = similarities.argsort()[::-1][:top_k]

        relevant_chunks = []
        for idx in top_indices:
            if similarities[idx] > 0.1:
                chunk = self.chunks[idx].copy()
                chunk["similarity_score"] = float(similarities[idx])
                relevant_chunks.append(chunk)
        return relevant_chunks

    def call_ollama(self, prompt: str) -> str:
        try:
            payload = {
                "model": self.llm_model_name,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "max_tokens": 200,
                }
            }
            response = requests.post(f"{self.ollama_base_url}/api/chat", json=payload, timeout=180)
            if response.status_code == 200:
                return response.json()["message"]["content"].strip()
            else:
                return f"Error: Ollama server returned status {response.status_code}"
        except requests.exceptions.ConnectionError:
            return "Error: Cannot connect to Ollama server. Please ensure Ollama is running."
        except requests.exceptions.Timeout:
            return "Error: Ollama request timed out. Please try again."
        except Exception as e:
            return f"Error: {str(e)}"

    def get_relevant_snippet(self, text: str, query: str, top_k=3) -> str:
        chunks = [chunk.strip() for chunk in text.split('\n\n') if chunk.strip()]
        chunk_embeddings = self.embedding_model.encode(chunks)
        query_embedding = self.embedding_model.encode([query])[0]
        similarities = cosine_similarity([query_embedding], chunk_embeddings)[0]
        top_indices = similarities.argsort()[::-1][:top_k]
        relevant_chunks = [chunks[i] for i in top_indices if similarities[i] > 0.1]
        if relevant_chunks:
            return "\n\n".join(relevant_chunks)
        return "Sorry, couldn't find relevant information."

    def chat(self, user_query: str) -> str:
        model_match = re.search(r'\b(P[Ss]\d+)\b', user_query)
        if model_match:
            model_number = model_match.group(1)
            try:
                part_data = self.scraper.get_part_information(model_number)
                part_info = part_data.get("part_info", {})
                description_text = part_info.get("description", "") or part_info.get("title", "")
                if description_text:
                    relevant_info = self.get_relevant_snippet(description_text, user_query)
                else:
                    relevant_info = "No detailed description available for this part."
                return f"Here is the relevant information for model {model_number}:\n\n{relevant_info}\n\nMore details: {part_data.get('url', 'URL not available')}"
            except WebDriverException as e:
                return f"Sorry, I failed to retrieve live info for model {model_number}: {e}"

        relevant_chunks = self.retrieve_relevant_chunks(user_query, top_k=3)
        seen_texts = set()
        context_parts = []
        for chunk in relevant_chunks:
            snippet = chunk["text"].strip()
            if snippet not in seen_texts:
                context_parts.append(snippet[:300])
                seen_texts.add(snippet)
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

You have access to the following function:
get_part_information(model_number: string) - searches for a model number and returns page content.
If you decide to call this function, your entire response MUST be exactly in this format:
[get_part_information(model_number="MODELNUMBER")]
Replace MODELNUMBER with the requested model number.
Do NOT include any other text if you call a function.
Answer:"""
        response = self.call_ollama(prompt)
        return response

    def get_stats(self) -> Dict:
        return {
            "documents": len(self.documents),
            "chunks": len(self.chunks),
            "embeddings_shape": self.embeddings.shape if self.embeddings is not None else None,
            "files": [doc["filename"] for doc in self.documents]
        }

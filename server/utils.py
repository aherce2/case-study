import re
from sklearn.metrics.pairwise import cosine_similarity

def chunk_text(text, chunk_size=500, overlap=50):
    sentences = re.split(r'[.!?]\s+', text)
    chunks = []
    current_chunk = ""
    overlap_text = ""
    chunk_id = 0
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
            current_chunk = overlap_text + " " + sentence
            chunk_id += 1
        else:
            current_chunk += " " + sentence if current_chunk else sentence

    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    return chunks

def get_top_k_similar_chunks(query_embedding, chunk_embeddings, chunks, top_k=5, min_score=0.1):
    similarities = cosine_similarity([query_embedding], chunk_embeddings)[0]
    top_indices = similarities.argsort()[::-1][:top_k]
    return [(chunks[i], similarities[i]) for i in top_indices if similarities[i] > min_score]

def sanitize_filename(name):
    return re.sub(r'[^a-zA-Z0-9_\-]', '_', name)

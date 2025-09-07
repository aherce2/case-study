from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agent import OllamaRAGAgent
from config import PRODUCTS, BASE_PROMPT, OLLAMA_BASE_URL, LLM_MODEL, EMBEDDING_MODEL, DOCS_FOLDER, VECTOR_DB_PATH

class Query(BaseModel):
    question: str

app = FastAPI()

# Allow frontend to connect in development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend dev server origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Initialize agent once at startup
agent = OllamaRAGAgent(
    products=PRODUCTS,
    system_prompt=BASE_PROMPT,
    docs_folder=DOCS_FOLDER,
    vector_db_path=VECTOR_DB_PATH,
    ollama_base_url=OLLAMA_BASE_URL,
    embedding_model_name=EMBEDDING_MODEL,
    llm_model_name=LLM_MODEL
)
agent.build_vector_index()

@app.post("/chat")
async def chat(query: Query):
    print(f"Query Recieved: {query.question}")
    response_text = agent.chat(query.question)  # agent processes the input
    return {"response": response_text}          # return JSON with 'response' field


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

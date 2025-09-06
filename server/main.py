from config import PRODUCTS, BASE_PROMPT, OLLAMA_BASE_URL, LLM_MODEL, EMBEDDING_MODEL, DOCS_FOLDER, VECTOR_DB_PATH
from server.agent import OllamaRAGAgent

def main():
    print("Initializing Agent...")
    agent = OllamaRAGAgent(
        products=PRODUCTS,
        system_prompt=BASE_PROMPT,
        docs_folder=DOCS_FOLDER,
        vector_db_path=VECTOR_DB_PATH,
        ollama_base_url=OLLAMA_BASE_URL,
        embedding_model_name=EMBEDDING_MODEL,
        llm_model_name=LLM_MODEL,
    )

    # Build or load existing vector index
    agent.build_vector_index()

    stats = agent.get_stats()
    print("\nKnowledge Base Stats:")
    print(f" Documents: {stats['documents']}")
    print(f" Text Chunks: {stats['chunks']}")
    print("Part Select Customer Service Agent ready!")
    print("Type 'exit' or 'quit' to end.\n")

    while True:
        try:
            user_input = input("You: ").strip()
            if user_input.lower() in ["exit", "quit"]:
                print("Thank you for contacting our service team!")
                break
            if not user_input:
                continue

            response = agent.chat(user_input)
            print(f"Agent: {response}\n")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}\n")

if __name__ == "__main__":
    main()

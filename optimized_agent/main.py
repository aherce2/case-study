from agent import PartSelectAgent
from config import LLM_MODEL
def main():
    print("Initializing Enhanced PartSelect Agent...")
    
    # Custom configuration example
    custom_config = {
        "model": {
            "LLM_MODEL_NAME": LLM_MODEL,
            "TEMPERATURE": 0.1
        },
        "processing": {
            "MAX_RETRIEVAL_CHUNKS": 5
        }
    }
    
    agent = PartSelectAgent(custom_config)
    
    # Build vector index
    print("Building knowledge base...")
    agent.build_index()
    
    print("PartSelect Customer Service Agent ready!")
    print("Commands: 'exit', 'history', 'clear', 'stats', 'conversations'\n")
    
    conversation_id = "main_session"
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if user_input.lower() == 'exit':
                print("Thank you for using PartSelect! 👋")
                break
            elif user_input.lower() == 'history':
                history = agent.get_conversation_history(conversation_id)
                print("\n--- Conversation History ---")
                for msg in history:
                    role = msg["role"].title()
                    content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
                    print(f"{role}: {content}")
                print("--- End History ---\n")
                continue
            elif user_input.lower() == 'clear':
                agent.clear_conversation(conversation_id)
                print("Conversation cleared!\n")
                continue
            elif user_input.lower() == 'stats':
                stats = agent.get_agent_stats()
                print("\n--- Agent Statistics ---")
                print(f"Model: {stats['config']['model_name']}")
                print(f"Documents: {stats['vector_db']['documents']}")
                print(f"Chunks: {stats['vector_db']['chunks']}")
                print(f"Total Conversations: {stats['conversations']['total_conversations']}")
                print(f"Cache Size: {stats['cache_size']}")
                print("--- End Statistics ---\n")
                continue
            elif user_input.lower() == 'conversations':
                conversations = agent.list_conversations()
                print(f"\n--- Available Conversations ({len(conversations)}) ---")
                for conv_id in conversations:
                    print(f"• {conv_id}")
                print("--- End Conversations ---\n")
                continue
            elif not user_input:
                continue
            
            response = agent.chat(user_input, conversation_id)
            print(f"\nAgent: {response}\n")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    main()

Building AI Agent Structure + Game Plan

* Define Agent’s Objective & Scope  
  * Focus the Agent: limit the task to assisting users about refrigerators and dishwasher part information, compatibility checks, installation, troubleshooting, and order support  
  * Define capabilities: agent should answer queries, guide users to part resources, validate part compatibility and escalate or clarify as needed  
* Set Up Core backend  
  * Environment: ensure secure environment variable management for API keys  
  * REST API Endpoint: endpoints for chat, data lookup and order management  
* Integrate Language model (deepseek)  
  * API Call: implement robust functions for sending prompts to deepseek and handle the responses  
  * Prompt engineering: carefully construct each request’s prompt to:  
    * Clearly specify the agent’s role and boundaries  
    * Provide relevant, concise product/data  
    * EG: “You are a product assistant for PartSelect. Only answer questions about refrigerators and dishwasher part selection, compatibility or installation. Politely decline unrelated topics”  
* Product knowledge Integration  
  * Product database hooks: allow the agent to query or look up product information as needed for accurate response  
    * Hardcode the case study  
    * In memory dictionaries, static JSON or vector database/document search for more advanced coverage  
  * Knowledge base: optionally incorporate an FAQ installation guides or troubleshooting  
* Context management & Conversation Logic  
  * Track conversation state: implement context tracking so agent knows what products, compatibility or installation steps have been discussed  
  * Clarification requests: when necessary have the agent ask for more information for precise recommendations or support  
* Response filtering & validation  
  * Guardrails: filter or validate model responses to block unsupported or out of scope topics  
  * Fallback escalation: when stumped, agent should admit limitations or offer to connect human support


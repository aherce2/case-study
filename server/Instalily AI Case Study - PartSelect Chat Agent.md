# Instalily AI Case Study \- PartSelect Chat

## **Summary**

This case study presents the development of an AI-powered chat agent for PartSelect's e-commerce platform, specifically focused on refrigerator and dishwasher parts. The implementation features a React-based frontend interface, a Python backend with web scraping capabilities, and integration with the Gemma 3 language model via Ollama for natural language processing.

## Technical Architecture

### Backend Implementation

#### **Language Model Selection**

* **Primary LM:** Gemma 3:latest via Ollama  
* **Rationale:** DeepSeek was initially considered but produced excessively long responses that impacted user experience. Gemma 3 provided more concise, focused responses suitable for customer service interactions.  
* **Flexibility:** The config.py file allows for easy LLM switching to any Ollama-supported model, ensuring future adaptability.

#### **Data Acquisition Strategy**

The core data pipeline revolves around a custom web scraping solution:

**scrape.py Class Features:**

* Selenium-based web scraping for structured data extraction  
* Configurable product categories: PRODUCTS \= \["refrigerator", "dishwasher"\]  
* Extensible design allowing easy addition of new product categories  
* Data persistence to part\_select\_data directory

**Design Decision \- Web Scraping vs. Browser Automation:** Initially planned to implement Browser Use for real-time website interaction, allowing the agent to dynamically navigate PartSelect's website. However, due to setup constraints within the 2-day timeline, I opted for a pre-scraping approach. This decision enabled development while maintaining data accuracy through PartSelect's structured URL patterns.

#### **Agent Architecture**

* **Current Implementation:** Single-agent system with comprehensive prompting  
* **Core Files:**  
  * test\_agent.py: Development testing interface for agent validation  
  * config.py: Centralized configuration for LLM settings and prompts

### Frontend Implementation

* **Framework:** React (modern, component-based architecture)  
* **Core Files:**  
  * config.py: Hidden File for localhost declaration

## Key Features & Capabilities

### **Supported Query Types**

1. **Part Installation Guidance:** "How can I install part number PS11752778?"  
2. **Troubleshooting Support:** "The ice maker on my Whirlpool fridge is not working. How can I fix it?"

### **Agent Specialization**

* **Scope Control:** Agent maintains focus on refrigerator and dishwasher parts  
* **Product Categories:** Easily extensible through PRODUCTS array modification

## Implementation Challenges & Solutions

### **Prompt Engineering Balance**

**Challenge:** Initial extensive and descriptive prompts limited the agent's ability to handle ambiguous queries, while overly simple prompts resulted in repetitive responses.

**Solution:** Iterative prompt refinement in config.py to find the optimal balance between specificity and flexibility. The current implementation provides consistent, relevant responses while maintaining some adaptability to user query variations.

## 

## Future Improvements & Roadmap

### **1\. Multi-Agent Architecture**

**Current Limitation:** Single-agent system handles all query types uniformly.

**Proposed Enhancement:** Implement a routing agent system with specialized sub-agents:

* **Troubleshooting Agent:** Focused on diagnostic and repair guidance  
* **Parts Inquiry Agent:** Specialized in product specifications and compatibility  
* **Installation Agent:** Dedicated to step-by-step installation procedures  
* **Order Support Agent:** Handles transaction-related queries

### **2\. Real-Time Website Integration**

**Future Goal:** Integrate Browser Use or similar tools to enable real-time website navigation and data retrieval.

**Benefits:**

* Always current product information  
* Dynamic inventory checking  
* Real-time pricing updates  
* Enhanced compatibility verification

### **3\. Enhanced Data Pipeline**

* **Vector Database Integration:** Implement vector storage for improved semantic search capabilities  
* **Automated Data Updates:** Scheduled scraping to maintain data freshness  
* **Multi-Source Integration:** Expand beyond web scraping to include API integrations where available

### **4\. Advanced Features**

* **Visual Product Recognition:** Image upload for part identification  
* **Interactive Installation Guides:** Step-by-step visual walkthroughs  
* **Predictive Maintenance:** Proactive part replacement suggestions  
* **Order Tracking Integration:** Real-time order status updates

## Technical Scalability Considerations

### **Horizontal Scaling**

* **Agent Architecture:** Modular design supports microservices migration  
* **Data Layer:** Scraping framework can distribute across multiple instances  
* **LLM Integration:** Ollama configuration allows for model load balancing

### **Performance Optimization**

* **Caching Strategy:** Implement Redis for frequently accessed product data  
* **Response Time:** Current local data approach ensures sub-second response times  
* **Resource Management:** Efficient memory usage through selective data loading

## Limitations & Constraints

### **Current Limitations**

1. **Static Data:** Reliance on pre-scraped data may result in outdated information  
2. **Single LLM:** No fallback language model for redundancy  
3. **Response Patterns:** Some repetitiveness in agent responses due to prompt optimization challenges

### **Mitigation Strategies**

* Regular data refresh schedules  
* Multi-model configuration options in config.py  
* Extensible product category system  
* Continuous prompt iteration and A/B testing

## **Conclusion**

**Key Achievements:**

* Functional chat interface with PartSelect branding alignment  
* Focused agent behavior for refrigerator and dishwasher parts  
* Extensible architecture supporting future enhancements  
* Comprehensive data scraping pipeline  
* Configurable LLM integration  
* Working demonstration of all requested query types

**Next Steps:**

1. Deploy multi-agent routing system  
2. Implement Browser Use for real-time data access  
3. Conduct user experience testing and optimization  
4. Expand product category support  
5. Integrate vector database for enhanced search capabilities

**Resources**

* [Mastering Caching LLM Calls with Tracing and Retrying](https://oleg-dubetcky.medium.com/mastering-caching-llm-calls-with-tracing-and-retrying-63e12c3318ef)  
*  [Building AI Agents In 44 Minutes](https://www.youtube.com/watch?v=_Udb5NC6vTI&t=1s)  
* [browser-use Github](https://github.com/browser-use/browser-use)  
* [RAG 101: Demystifying Retrieval-Augmented Generation Pipelines](https://developer.nvidia.com/blog/rag-101-demystifying-retrieval-augmented-generation-pipelines/)


  

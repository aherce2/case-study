# Instalily AI Case Study \- PartSelect Chat

## **Summary**

This case study presents a significantly enhanced AI-powered chat agent for PartSelect's e-commerce platform, featuring advanced RAG (Retrieval-Augmented Generation) architecture, intelligent function calling, persistent conversation management, and semantic caching. The implementation has evolved from a basic single-agent system to a solution with modular design and improved scalability.

## Major Architectural Improvements

### **1\. Advanced RAG Architecture**

**Previous Implementation**: Simple web scraping with basic text retrieval 

**Modified Implementation**: Vector database system with semantic search

#### **Core Components:**

* **Vector Database (agent.py \- VectorDatabase class)**: Persistent storage of document embeddings with metadata  
* **Document Processing (agent.py \- DocumentProcessor class)**: Advanced chunking with overlap and metadata extraction  
* **Semantic Retrieval**: Cosine similarity-based relevant chunk retrieval with configurable thresholds  
* **Embedding Model**: SentenceTransformer integration for high-quality semantic understanding

### **2\. Intelligent Function Calling System**

**New Implementation**: Complete function calling framework replacing simple pattern matching

#### **Function Architecture (handler.py):**

* **Base Function Handler**: Abstract base class for extensible function system  
* **Specialized Handlers**:  
  * GetPartInformationHandler: Real-time part information retrieval  
  * CheckCompatibilityHandler: Model-part compatibility verification  
* **Function Parser**: Advanced regex patterns for multiple function call formats  
* **Result Management**: Structured result objects with error handling and metadata

#### **Function Call Examples:**

* \[get\_part\_information(model\_number="PS11752778")\]  
* \[check\_model\_part\_compatibility(model\_number="WDT780SAEM1", part\_number="PS11752778")\]

### **3\. Persistent Conversation Management**

**New Feature**: Complete conversation persistence and context tracking

#### **Conversation System (conversation\_manager.py):**

* **Message Storage**: JSON-based persistent storage with metadata  
* **Entity Extraction**: Automatic part/model number detection using regex patterns  
* **Context Resolution**: Reference resolution ("this part", "my model") using conversation history  
* **Memory Management**: In-memory caching with periodic cleanup  
* **History Management**: Configurable conversation history limits

#### **Entity Patterns (config.py \- EntityPatterns):**

* Part Numbers: PS\\d{8,}, WP\[A-Z0-9\]{8,}, etc.  
* Model Numbers: \[A-Z\]{3,4}\\d{6,12}\[A-Z\]?\\d{0,2}, WDT\\w+, etc.

### **4\. Semantic Caching System**

**New Feature**: Advanced LLM response caching with semantic similarity

#### **Cache Implementation (semantic\_cache.py):**

* **Disk Cache**: Persistent storage using diskcache  
* **Semantic Matching**: Cosine similarity for cache hit detection  
* **Retry Logic**: Tenacity-based retry with exponential backoff  
* **Embedding-Based Keys**: Cache keys based on semantic embeddings rather than exact text matches

### **5\. Modular Configuration System**

**Enhanced Configuration (config.py)**:

#### **Configuration Classes:**

* **DatabaseConfig**: Storage paths and directories  
* **ModelConfig**: LLM and embedding model settings  
* **ProcessingConfig**: Text chunking and retrieval parameters  
* **ScrapingConfig**: Web scraping configuration with retry logic  
* **ConversationConfig**: Chat history and context management  
* **SecurityConfig**: Rate limiting and content filtering

#### **System Prompts:**

* **Enhanced Prompts**: Structured prompt templates with conversation context  
* **Function Integration**: Clear function calling instructions  
* **Context Templates**: Dynamic context injection for better responses

## **Implementation Details**

### **File Structure and Responsibilities**

#### **Core Agent (agent.py)**

* **PartSelectAgent**: Main orchestration class with custom configuration support  
* **VectorDatabase**: Persistent vector storage with save/load capabilities  
* **DocumentProcessor**: Enhanced text chunking with metadata preservation  
* **LLMService**: Ollama integration with conversation context and caching

#### **Enhanced Web Scraping (scraper.py)**

* **Improved Error Handling**: Comprehensive exception management  
* **Structured Data Models**: ScrapingResult and PartInfo dataclasses  
* **Retry Logic**: Configurable retry mechanisms for reliability  
* **CSS Selectors**: Centralized selector management in config.py

#### **Utility Functions (utils.py)**

* **Text Processing**: Advanced chunking with sentence boundary detection  
* **Entity Validation**: Part/model number validation functions  
* **File Management**: Comprehensive file operations and metadata extraction  
* **Performance Utilities**: Batch processing and text similarity measures

### **Advanced Features**

#### **1\. Reference Resolution**

The conversation manager automatically resolves pronouns and references:

* "this part" → specific part number from context  
* "my model" → user's appliance model from conversation history  
* Context-aware entity tracking across conversation turns

#### **2\. Intelligent Chunking**

Document processing with improved chunking strategy:

* Sentence boundary detection for better chunk coherence  
* Configurable overlap for context preservation  
* Metadata preservation including document structure

#### **3\. Multi-Level Caching**

Three-tier caching system:

* **Query Cache**: Exact query matches for instant responses  
* **Semantic Cache**: Similarity-based cache for related queries  
* **Vector Cache**: Persistent embedding storage

#### **4\. Comprehensive Error Handling**

* Graceful degradation for service failures  
* Detailed error logging with context  
* User-friendly error messages  
* Automatic retry mechanisms

## **Performance and Scalability Improvements**

### **1\. Memory Management**

* In-memory conversation cache with configurable limits  
* Periodic cleanup of old conversations  
* Efficient vector storage with numpy arrays  
* Lazy loading of embeddings and documents

### **2\. Response Time Optimization**

* Sub-second response times through local vector database  
* Semantic caching reduces LLM calls  
* Efficient similarity calculations using sklearn  
* Parallel processing capabilities for batch operations

### **3\. Extensibility Framework**

* Plugin-based function handler system  
* Configurable product categories  
* Modular component architecture  
* Environment-based configuration overrides

## **Migration from Original Implementation**

### **Deprecated Components**

* **Simple Pattern Matching**: Replaced with intelligent function calling  
* **Basic Text Search**: Upgraded to semantic vector search  
* **Stateless Interactions**: Enhanced with persistent conversation management  
* **Single-File Architecture**: Modularized into specialized components

## **Deployment and Production Readiness**

### **Configuration Management**

custom\_config \= {  
    "model": {  
        "LLM\_MODEL\_NAME": "gemma3:latest",  
        "TEMPERATURE": 0.1  
    },  
    "processing": {  
        "MAX\_RETRIEVAL\_CHUNKS": 5,  
        "SIMILARITY\_THRESHOLD": 0.1  
    }  
}

### **Monitoring and Observability**

* Comprehensive logging with structured output  
* Performance metrics collection  
* Error tracking and reporting  
* Usage statistics and analytics

## **Future Improvements**

### **1\. Multi-Agent Architecture**

* **Agent Routing**: Intelligent query routing to specialized sub-agents  
* **Domain Experts**: Separate agents for troubleshooting, installation, and parts inquiry  
* **Orchestration Layer**: Central coordinator for complex multi-step interactions

### **2\. Advanced AI Features**

* **Vision Integration**: Image-based part identification  
* **Predictive Analytics**: Proactive maintenance suggestions  
* **Learning System**: Continuous improvement from user interactions  
* **Multi-Modal Responses**: Rich media responses with images and videos

## **Technical Metrics and Performance**

### **System Capabilities**

* **Document Processing**: Handles thousands of documents with efficient chunking  
* **Response Time**: Sub-second responses for cached queries, 2-3 seconds for new queries  
* **Accuracy**: Improved retrieval accuracy through semantic search  
* **Scalability**: Horizontal scaling support through modular architecture

### **Resource Requirements**

* **Memory**: Optimized memory usage with configurable limits  
* **Storage**: Persistent vector database with incremental updates  
* **CPU**: Efficient embedding calculations with GPU acceleration support  
* **Network**: Minimal API calls through intelligent caching

## **Conclusion**

### **Key Achievements:**

* **Scalability**: Vector database and caching systems support thousands of concurrent users  
* **Extensibility**: Plugin architecture enables rapid feature development  
* **Performance**: Significant improvement in response quality and relevance

### **Technical Improvements:**

* **Clean Architecture**: Separation of concerns with clear component boundaries  
* **Type Safety**: Comprehensive dataclass usage for better code reliability  
* **Error Resilience**: Graceful degradation and comprehensive error handling  
* **Configuration Management**: Environment-aware configuration with override capabilities  
* **Testing Infrastructure**: Modular design enables unit testing


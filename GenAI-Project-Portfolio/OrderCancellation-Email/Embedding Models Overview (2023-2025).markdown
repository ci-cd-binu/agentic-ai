# Embedding Models: 2023–2025
*An Overview for GenAI Architects*

## Introduction
Embedding models convert text, images, or other data into dense vector representations, enabling tasks like semantic search, clustering, and RAG. This document covers key embedding models from 2023 to 2025, categorized by open-source and commercial offerings, with relevance for GenAI applications.

---

## Key Trends in Embedding Models
- **Multimodal Embeddings**: Models supporting text, images, and audio for versatile applications.
- **Efficiency**: Lightweight models optimized for low-latency inference and edge deployment.
- **Open-Source Growth**: Increased availability of high-quality open-source embeddings.
- **Scalability**: Models designed for large-scale enterprise search and retrieval.

---

## Commercial Embedding Models

### OpenAI text-embedding-3-large
- **Developer**: OpenAI
- **Launch**: January 2024
- **License**: Commercial
- **Dimension**: 3072
- **Key Features**: High-dimensional embeddings, optimized for semantic search and clustering.
- **Use Cases**: Enterprise search, RAG, text classification.
- **Relevance**: High accuracy for English-centric tasks; accessible via OpenAI’s API.
- **Hands-On Insight**: Integrate with OpenAI’s API for RAG pipelines, using vector databases like Pinecone.

### Cohere Embed v3
- **Developer**: Cohere
- **Launch**: November 2023
- **License**: Commercial
- **Dimension**: 1024
- **Key Features**: Multilingual support, optimized for low-latency retrieval.
- **Use Cases**: Knowledge retrieval, multilingual search, classification.
- **Relevance**: Ideal for enterprise applications requiring fast, multilingual embeddings.
- **Hands-On Insight**: Use Cohere’s API with Weaviate for scalable RAG systems.

### Google Vertex AI Embeddings
- **Developer**: Google
- **Launch**: March 2023
- **License**: Commercial
- **Dimension**: 768
- **Key Features**: Integrated with Google Cloud, supports text and multimodal embeddings.
- **Use Cases**: Enterprise search, recommendation systems, multimodal RAG.
- **Relevance**: Seamless integration with Google’s ecosystem for large-scale deployments.
- **Hands-On Insight**: Deploy on Vertex AI, combining with BigQuery for data-driven applications.

---

## Open-Source Embedding Models

### Sentence-BERT (SBERT)
- **Developer**: UKP Lab (Hugging Face Community)
- **Launch**: Ongoing updates (2023–2025)
- **License**: Apache 2.0
- **Dimension**: 384–768
- **Key Features**: Lightweight, fine-tunable, supports multiple languages.
- **Use Cases**: Semantic search, text similarity, clustering.
- **Relevance**: Highly customizable for domain-specific tasks; widely adopted in research.
- **Hands-On Insight**: Fine-tune SBERT with Hugging Face’s sentence-transformers library for niche applications.

### BGE (BGE-large-en-v1.5)
- **Developer**: BAAI (Beijing Academy of AI)
- **Launch**: September 2023
- **License**: Apache 2.0
- **Dimension**: 1024
- **Key Features**: State-of-the-art for English text embeddings, optimized for retrieval.
- **Use Cases**: RAG, semantic search, question answering.
- **Relevance**: Competitive with commercial models, ideal for open-source RAG pipelines.
- **Hands-On Insight**: Deploy BGE with Hugging Face’s Transformers, integrating with Faiss for vector search.

### E5 (intfloat/e5-large-v2)
- **Developer**: Microsoft (Community-driven)
- **Launch**: December 2023
- **License**: MIT
- **Dimension**: 1024
- **Key Features**: High performance for text and code embeddings, multilingual support.
- **Use Cases**: Code search, document retrieval, multilingual RAG.
- **Relevance**: Versatile for cross-domain applications, especially in technical domains.
- **Hands-On Insight**: Use E5 with ONNX for efficient inference in production.

### Mixtral Embed
- **Developer**: Mistral AI
- **Launch**: February 2024
- **License**: Apache 2.0
- **Dimension**: 768
- **Key Features**: Lightweight, MoE-based embeddings, optimized for low-GPU usage.
- **Use Cases**: Edge applications, cost-effective retrieval.
- **Relevance**: Ideal for resource-constrained environments.
- **Hands-On Insight**: Deploy Mixtral Embed with vLLM for low-latency inference.

### CLIP-ViT-L-336px
- **Developer**: OpenAI
- **Launch**: June 2023
- **License**: MIT
- **Dimension**: 512
- **Key Features**: Multimodal (text and image) embeddings, high accuracy for vision tasks.
- **Use Cases**: Image-text retrieval, multimodal search, content moderation.
- **Relevance**: Pioneering multimodal embeddings for cross-modal applications.
- **Hands-On Insight**: Use CLIP with PyTorch for multimodal RAG pipelines.

---

## Embedding Model Cheat Sheet (2025)

| Model                     | Company                       | License         | Dimension | Open-Source? | Key Use Case                     |
|---------------------------|-------------------------------|-----------------|-----------|--------------|----------------------------------|
| text-embedding-3-large    | OpenAI                        | Commercial      | 3072      | No           | Enterprise search, RAG           |
| Cohere Embed v3           | Cohere                        | Commercial      | 1024      | No           | Multilingual retrieval, Q&A       |
| Google Vertex AI          | Google                        | Commercial      | 768       | No           | Multimodal RAG, enterprise search|
| SBERT                     | UKP Lab (Hugging Face)        | Apache 2.0      | 384–768   | Yes          | Semantic search, clustering      |
| BGE-large-en-v1.5         | BAAI                          | Apache 2.0      | 1024      | Yes          | RAG, question answering          |
| E5-large-v2               | Microsoft                     | MIT             | 1024      | Yes          | Code search, multilingual RAG    |
| Mixtral Embed             | Mistral AI                    | Apache 2.0      | 768       | Yes          | Edge applications, retrieval      |
| CLIP-ViT-L-336px         | OpenAI                        | MIT             | 512       | Yes          | Multimodal search, image retrieval|

---

## Relevance for GenAI Architects
- **Commercial Models**: Offer high accuracy and enterprise support but require API costs and vendor lock-in. Ideal for rapid deployment in production.
- **Open-Source Models**: Provide flexibility for fine-tuning and cost-effective deployment, especially for research or custom applications.
- **Hands-On Skills**:
  - Fine-tune SBERT or BGE for domain-specific embeddings using Hugging Face.
  - Build RAG pipelines with open-source embeddings (e.g., BGE) and vector databases (e.g., Faiss, Pinecone).
  - Optimize multimodal embeddings (e.g., CLIP) for cross-modal search using PyTorch.
  - Monitor embedding performance with metrics like cosine similarity and recall@K.

---

## Future Trends in Embedding Models
1. **Multimodal Integration**: Unified embeddings for text, images, and audio.
2. **Efficiency**: Lightweight models for edge and on-device applications.
3. **Open-Source Adoption**: Growing community contributions to models like BGE and E5.
4. **Scalability**: Optimized embeddings for large-scale retrieval in enterprise settings.
5. **Ethical Considerations**: Addressing biases in embeddings for fair representations.

**Hands-On Insight**: Experiment with multimodal embeddings (e.g., CLIP) for innovative applications like visual question answering, and use tools like Faiss for scalable vector search.
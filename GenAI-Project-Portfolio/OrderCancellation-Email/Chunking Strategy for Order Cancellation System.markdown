# Chunking Strategy for Order Cancellation System

## 1. Overview
Chunking involves breaking large documents into smaller segments for efficient retrieval and processing in RAG systems. Effective chunking balances granularity, context preservation, and retrieval performance. Common strategies include fixed-size, semantic, and recursive character chunking.

## 2. Application in Use Case
The order cancellation system chunked documents (FAQs, glossaries, emails) for storage in Qdrant and retrieval in the RAG pipeline. The goal was to ensure:
- **Relevance**: Retrieved chunks contained complete, relevant information.
- **Efficiency**: Minimized storage and retrieval overhead.
- **Context**: Preserved enough context for LLM understanding.

### Chunking Strategy
- **Method**: Semantic chunking based on sentence boundaries and topic coherence.
- **Rationale**:
  - Fixed-size chunking risked splitting sentences, losing context.
  - Semantic chunking grouped related sentences, improving retrieval relevance.

## 3. Technical Details
- **Pipeline**:
  - Preprocessing: Split documents into sentences using spaCy.
  - Embedding: Generated sentence embeddings with `text-embedding-gecko`.
  - Clustering: Grouped sentences into chunks (max 5 sentences) based on cosine similarity (>0.8).
  - Storage: Stored chunks in Qdrant with metadata (e.g., document_id, topic).
    ```python
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import AgglomerativeClustering

    model = SentenceTransformer('all-MiniLM-L6-v2')
    sentences = ['sentence1', 'sentence2', ...]
    embeddings = model.encode(sentences)
    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=0.2)
    labels = clustering.fit_predict(embeddings)
    chunks = [sentences[i] for i in range(len(labels)) if labels[i] == cluster_id]
    ```
- **Chunk Size**:
  - Average: 3-5 sentences, ~100-200 tokens.
  - Max: 512 tokens to fit LLM context window.
- **Performance**:
  - Retrieval precision: 0.90 (RAGAS).
  - Storage: 10,000 chunks for 1,000 documents.

## 4. Challenges and Mitigations
- **Challenge**: Over-segmentation losing context.
  - **Mitigation**: Ensured minimum chunk size of 2 sentences.
- **Challenge**: High computational cost for semantic chunking.
  - **Mitigation**: Precomputed embeddings offline using Dataflow.
- **Challenge**: Inconsistent chunk relevance.
  - **Mitigation**: Used metadata filtering (e.g., topic) during retrieval.

## 5. Interview Readiness
- **Key Points**:
  - Compare chunking strategies (fixed-size, semantic, recursive).[](https://www.analyticsvidhya.com/blog/2024/10/learning-path-for-ai-agents/)
  - Discuss trade-offs between chunk size and retrieval performance.
  - Explain how chunking impacts RAG quality and LLM context.
- **Recent Trends**:
  - Semantic chunking with LLMs for topic detection.
  - Dynamic chunking based on query complexity.
  - Multimodal chunking for text and images.[](https://www.analyticsvidhya.com/blog/2024/11/generative-ai-interview-questions/)
- **Articulation Tips**:
  - Highlight how semantic chunking improved RAG relevance.
  - Discuss optimization for production (e.g., offline embedding).
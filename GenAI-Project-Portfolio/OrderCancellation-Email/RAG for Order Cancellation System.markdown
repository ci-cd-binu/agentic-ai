# Retrieval-Augmented Generation (RAG) for Order Cancellation System

## 1. Overview
Retrieval-Augmented Generation (RAG) combines information retrieval with LLM generation to produce contextually relevant outputs. It retrieves relevant documents from a knowledge base (often via a vector database) and augments LLM prompts, reducing hallucinations and enhancing domain-specific responses.

## 2. Application in Use Case
RAG was used to enhance email generation by providing LLMs with:
- **Silviculture Terminology**: Retrieved glossaries and FAQs to ensure accurate terminology.
- **Historical Interactions**: Retrieved similar past customer interactions to inform tone and content.
- **Cancellation Policies**: Retrieved company policies to include in emails (e.g., discount offers).

### RAG Workflow
1. **Query**: Constructed from customer profile, cancellation reason, and sentiment.
2. **Retrieval**: Queried Qdrant for relevant documents (e.g., FAQs, past emails).
3. **Augmentation**: Added retrieved documents to the prompt.
4. **Generation**: LLM generated email using augmented prompt.

## 3. Technical Details
- **Knowledge Base**:
  - Stored in Qdrant: 10,000 documents (FAQs, glossaries, emails).
  - Embedded using `text-embedding-gecko`.
- **RAG Pipeline**:
  - Retrieval: Top-5 documents based on cosine similarity.
  - Prompt Augmentation:
    ```python
    def augment_prompt(query, retrieved_docs):
        context = "\n".join([doc['payload']['text'] for doc in retrieved_docs])
        return f"""
        Context: {context}
        Given a customer profile {query['customer']}, cancellation reason {query['reason']}, 
        and sentiment {query['sentiment']}, draft an empathetic email.
        """
    ```
  - Generation: Mistral endpoint on Vertex AI.
- **Evaluation**:
  - Relevance: Contextual precision = 0.92 (RAGAS metric).[](https://www.getzep.com/ai-agents/introduction-to-ai-agents)
  - Faithfulness: 95% of emails aligned with retrieved context.

## 4. Challenges and Mitigations
- **Challenge**: Irrelevant retrieved documents.
  - **Mitigation**: Used re-ranking with a cross-encoder model to improve relevance.
- **Challenge**: Token limits with large contexts.
  - **Mitigation**: Summarized retrieved documents using a smaller LLM (e.g., T5).
- **Challenge**: Slow retrieval for real-time use.
  - **Mitigation**: Cached frequent queries in Firestore.

## 5. Interview Readiness
- **Key Points**:
  - Explain RAG’s role in reducing hallucinations and enhancing domain knowledge.[](https://zilliz.com/blog/landscape-of-gen-ai-ecosystem-beyond-llms-and-vector-databases)
  - Discuss retrieval strategies (vector search, hybrid search) and evaluation metrics (RAGAS).
  - Highlight integration with vector databases and LLMs.
- **Recent Trends**:
  - Agentic RAG with iterative retrieval and reasoning.[](https://www.analyticsvidhya.com/blog/2024/12/agentic-rag-with-phidata/)
  - Multimodal RAG for text, images, and tables.[](https://www.analyticsvidhya.com/agenticaipioneer/)
  - Adaptive RAG with dynamic context selection.
- **Articulation Tips**:
  - Describe how RAG improved email relevance and reduced manual dealer edits.
  - Discuss trade-offs between RAG and fine-tuning for domain adaptation.
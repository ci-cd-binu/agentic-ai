### Design and develop a RAG solution
01/10/2025
The Retrieval-Augmented Generation (RAG) pattern is an industry-standard approach to building applications that use language models to process specific or proprietary data that the model doesn't already know. The architecture is straightforward, but designing, experimenting with, and evaluating RAG solutions that fit into this architecture involve many complex considerations that benefit from a rigorous, scientific approach.

This article is the introduction of a series. Each article in the series covers a specific phase in RAG solution design.

The other articles in this series cover the following considerations:

  How to determine which test documents and queries to use during evaluation
  How to choose a chunking strategy
  How to determine which chunks you should enrich and how to enrich them
  How to choose the right embedding model
  How to configure the search index
  How to determine which searches, such as vector, full text, hybrid, and manual multiple searches, you should perform
  How to evaluate each step
### RAG architecture
Diagram that shows the high-level architecture of a RAG solution, including the request flow and the data pipeline.
<img width="1669" height="506" alt="image" src="https://github.com/user-attachments/assets/92adac20-2943-40af-bfc9-08e3fb706c9a" />



### RAG application flow
The following workflow describes a high-level flow for a RAG application.

The user issues a query in an intelligent application user interface.
The intelligent application makes an API call to an orchestrator. You can implement the orchestrator with tools or platforms like the Microsoft Agent Framework, Semantic Kernel, Azure AI Agent service, or LangChain.
The orchestrator determines which search to perform on Azure AI Search and issues the query.
The orchestrator packages the top N results from the query. It packages the top results and the query as context within a prompt and sends the prompt to the language model. The orchestrator returns the response to the intelligent application for the user to read.
RAG data pipeline flow
The following workflow describes a high-level flow for a data pipeline that supplies grounding data for a RAG application.

### Documents are either pushed or pulled into a data pipeline.
The data pipeline processes each document individually by completing the following steps:
Chunk document: Breaks down the document into semantically relevant parts that ideally have a single idea or concept.
Enrich chunks: Adds metadata fields that the pipeline creates based on the content in the chunks. The data pipeline categorizes the metadata into discrete fields, such as title, summary, and keywords.
Embed chunks: Uses an embedding model to vectorize the chunk and any other metadata fields that are used for vector searches.
Persist chunks: Stores the chunks in the search index.
RAG design and evaluation considerations
You must make various implementation decisions as you design your RAG solution. The following diagram illustrates some of the questions you should ask when you make those decisions.

### Diagram that shows the high-level architecture of a RAG solution, including the questions that you should ask as you design the solution.
<img width="1492" height="664" alt="image" src="https://github.com/user-attachments/assets/e7778850-4486-4738-bd86-90dcd21ef65b" />
The following list provides a brief description of what you should do during each phase of RAG solution development.

During the preparation phase, you should:

Determine the solution domain. Clearly define the business requirements for the RAG solution.
Gather representative test documents. Gather test documents for your RAG solution that are representative of your document collection.
Gather test queries. Gather information and test queries and generate synthetic queries and queries that your documents don't cover.
During the chunking phase, you should:

Understand chunking economics. Understand which factors to consider as you evaluate the overall cost of your chunking solution for your text collection.
Perform document analysis. Ask the following questions to help you make decisions when you analyze a document type:
What content in the document do you want to ignore or exclude?
What content do you want to capture in chunks?
How do you want to chunk that content?
Understand chunking approaches. Understand the different approaches to chunking, including sentence-based, fixed-size, and custom approaches or by using language model augmentation, document layout analysis, and machine learning models.
Understand how document structure affects chunking. Choose a chunking approach based on the degree of structure that the document has.
During the chunk enrichment phase, you should:

Clean chunks. Implement cleaning approaches to eliminate differences that don't affect the meaning of the text. This method supports closeness matches.
Augment chunks. Consider augmenting your chunk data with common metadata fields and understand their potential uses in search. Learn about commonly used tools or techniques for generating metadata content.
During the embedding phase, you should:

Understand the importance of the embedding model. An embedding model can significantly affect the relevancy of your vector search results.
Choose the right embedding model for your use case.
Evaluate embedding models. Evaluate embedding models by visualizing embeddings and calculating embedding distances.
During the information retrieval phase, you should:

Create a search index. Apply the appropriate vector search configurations to your vector fields.
Understand search options. Consider the different types of searches, including vector, full-text, hybrid, and manual multiple searches. Learn about how to split a query into subqueries and filter queries.
Evaluate searches. Use retrieval evaluation methods to evaluate your search solution.
During the language model end-to-end evaluation phase, you should:

Understand language model evaluation metrics. There are several metrics, including groundedness, completeness, utilization, and relevancy, that you can use to evaluate the language model's response.
Understand similarity and evaluation metrics. You can use similarity and evaluation metrics to evaluate your RAG solution.
Understand the importance of documentation, reporting, and aggregation. Document the hyperparameters and the evaluation results. Aggregate the results from multiple queries and visualize the results.
Use the RAG experiment accelerator. Use the RAG experiment accelerator GitHub repository to help your team find the best strategies for RAG implementation by running multiple experiments, persisting, and evaluating the results.
Structured approach
Because of the number of steps and variables, it's important that you follow a structured evaluation process for your RAG solution. Evaluate the results of each step and make changes based on your requirements. You should evaluate each step independently for optimization, but remember that the end result is what your customers experience. Make sure that you understand all of the steps in this process before you determine your own acceptance criteria for each step.

Contributors
This article is maintained by Microsoft. It was originally written by the following contributors.

Principal authors:

Raouf Aliouat | Software Engineer II
Rob Bagby | Principal Content Developer - Azure Patterns & Practices
Prabal Deb | Principal Software Engineer
Chad Kittel | Principal Software Engineer - Azure Patterns & Practices
Ritesh Modi | Principal Engineer
Ryan Pfalz | Senior Technical Program Manager
Randy Thurman | Principal AI Cloud Solution Architect
Perfect — you’re asking for **a deep-dive explanation of every core RAG component** written at the same level of detail as your Retriever section.

Below is a full **RAG architecture component guide** designed for **interview or architectural discussions** — clear, layered, and example-backed.

---

# 🧠 Retrieval-Augmented Generation (RAG) — Component-Level Deep Dive

Retrieval-Augmented Generation (RAG) is a pipeline architecture that enhances an LLM’s reasoning by grounding it on external factual knowledge.
Let’s walk through each major component — **Controller**, **Retriever**, **Embedding Model**, **Vector Store**, and **LLM** — in depth.

---

## 🧩 1️⃣ Controller (Orchestrator / Manager Agent)

The **Controller** is the **central coordinator** in a RAG or agentic pipeline.

It decides **how** and **when** to call other components (retriever, tools, LLMs) and merges their outputs into a coherent final response.

---

### 🔹 Role in RAG

The Controller receives a user’s question and orchestrates the RAG workflow:

```
User Query → Controller → Retriever → Context → LLM → Final Response
```

It ensures the **query is preprocessed, grounded, and validated** before passing it to the model.

---

### 🔹 Step-by-Step Inside the Controller

1️⃣ **Receive and Classify Query**

The controller interprets user intent:

* Is it factual (→ needs retrieval)?
* Is it analytical (→ LLM reasoning)?
* Or a combination (→ retrieval + synthesis)?

2️⃣ **Trigger Retrieval (if factual)**

It formulates a **search query** or vector embedding request and invokes the Retriever to get relevant chunks.

3️⃣ **Context Assembly**

It merges the original user question with the retrieved text snippets into a structured **prompt**.

Example prompt template:

```
Answer based on the context below:
<context>
...
</context>

User Question: ...
```

4️⃣ **Safety & Policy Enforcement**

Controllers often apply:

* Guardrails (e.g., to filter sensitive topics)
* Session state or memory retrieval
* Post-processing (re-ranking, summarizing, truncating long contexts)

5️⃣ **Send to LLM**

The final composed input (user query + retrieved context) is sent to the LLM to generate the grounded answer.

---

### 🔹 Controller in Practice

| Framework          | Controller Equivalent                                      |
| ------------------ | ---------------------------------------------------------- |
| **LangChain**      | `Chain`, `AgentExecutor`, or custom Router Chain           |
| **Google ADK**     | Root Agent (delegates to retriever & reasoning sub-agents) |
| **NVIDIA NeMo**    | RAG Pipeline Orchestrator                                  |
| **OpenAI RAG API** | Implicit Controller within Retrieval Plugin                |

---

### 🔹 Summary — Controller Responsibilities

| Stage | Function                            |
| ----- | ----------------------------------- |
| 🧩 1  | Interpret user query                |
| 🧩 2  | Decide retrieval necessity          |
| 🧩 3  | Call retriever & preprocess context |
| 🧩 4  | Assemble prompt                     |
| 🧩 5  | Call LLM & finalize response        |

**In short:**
The Controller acts as the “conductor of the RAG orchestra,” ensuring each component plays in harmony.

---

## 🧩 2️⃣ Retriever — The Semantic Search Engine of RAG

The **Retriever** locates relevant knowledge chunks that ground the LLM’s response.

It converts the query into an embedding and searches a **vector database** for semantically similar data points.

---

### 🔹 Step-by-Step Inside the Retriever

1️⃣ **Embed the Query**

Convert query → embedding vector via an **embedding model**.

Example:

```
“What is the refund policy for premium customers?”
→ [0.02, -0.14, 0.88, ...]
```

2️⃣ **Vector Similarity Search**

Compare query vector with stored document embeddings in the vector DB using cosine similarity or dot product.

3️⃣ **Return Top-K Context**

Return the most relevant K chunks + metadata for the Controller to use.

---

### 🔹 Summary — Retriever Role

| Stage | Component         | Task                      |
| ----- | ----------------- | ------------------------- |
| 1️⃣   | Embedding Model   | Convert query → vector    |
| 2️⃣   | Vector DB / Index | Compute similarity        |
| 3️⃣   | Retriever         | Return Top-K context      |
| 4️⃣   | Controller        | Pass to LLM for reasoning |

---

## 🧩 3️⃣ Embedding Model — The Semantic Translator

The **Embedding Model** transforms text into numerical vectors that capture meaning and similarity.

---

### 🔹 Role in RAG

It’s the **bridge** between raw language and mathematical similarity search.

Similar text → similar embeddings.
Example:

* “refund policy for gold members”
* “premium plan reimbursements”
  These will have high cosine similarity (≈ 0.9).

---

### 🔹 Popular Embedding Models

| Vendor     | Model                    | Dimensions | Notes                                      |
| ---------- | ------------------------ | ---------- | ------------------------------------------ |
| **Google** | `text-embedding-004`     | 768        | Excellent semantic coherence               |
| **OpenAI** | `text-embedding-3-large` | 3072       | Multilingual, precise                      |
| **BAAI**   | `bge-large-en`           | 1024       | Open-source, strong English model          |
| **Cohere** | `embed-english-v3.0`     | 1024       | Domain-tunable                             |
| **NVIDIA** | `NV-Embed-QA`            | 1024       | Optimized for retrieval in NeMo Guardrails |

---

### 🔹 Design Considerations

* Choose models **aligned to your domain** (e.g., legal, medical, finance)
* Maintain **embedding-version consistency** across index and retriever
* Normalize vectors (unit length) before indexing

---

### 🔹 Example Use (LangChain)

```python
from langchain.embeddings import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector = embeddings.embed_query("refund policy for premium users")
```

---

## 🧩 4️⃣ Vector Database (Index Store)

The **Vector DB** stores embeddings and supports fast **nearest neighbor search** to retrieve relevant context.

---

### 🔹 Common Options

| Tool                          | Type              | Key Features                     |
| ----------------------------- | ----------------- | -------------------------------- |
| **Pinecone**                  | SaaS              | Managed vector search, scalable  |
| **FAISS**                     | Local library     | Fast CPU/GPU indexing            |
| **Weaviate**                  | Open-source DB    | Hybrid search (vector + keyword) |
| **Chroma**                    | Lightweight local | Great for prototyping            |
| **Vertex AI Matching Engine** | GCP-native        | Production-grade at scale        |

---

### 🔹 Search Process

1️⃣ User query → embedding
2️⃣ Vector DB computes similarity with stored vectors
3️⃣ Returns top-K chunks with metadata (e.g., document title, page number)

Example output:

```json
[
  {"chunk": "Refund policy applies to premium users...", "score": 0.91},
  {"chunk": "Cancellation window is 30 days...", "score": 0.86}
]
```

---

### 🔹 Performance Tuning

* Index type: IVF, HNSW, FlatL2 (depending on scale)
* Tradeoff between speed and precision (recall)
* Use metadata filters (e.g., doc_type="policy") for structured search

---

## 🧩 5️⃣ LLM (Reasoning and Response Generator)

The **Large Language Model** synthesizes the final answer using the query + retrieved context.

---

### 🔹 Step-by-Step Inside the LLM Stage

1️⃣ **Input Construction**

The Controller sends a prompt like:

```
Context:
[retrieved text 1]
[retrieved text 2]

Question: What is the refund policy for premium customers?
```

2️⃣ **Grounded Generation**

The LLM conditions its output on this context — “retrieval-augmented reasoning.”

3️⃣ **Response Validation**

Modern frameworks use guardrails or re-ranking to ensure the LLM doesn’t hallucinate beyond retrieved context.

---

### 🔹 Typical LLMs Used in RAG

| Vendor                | Model                  | Notes                             |
| --------------------- | ---------------------- | --------------------------------- |
| **Google**            | Gemini 1.5 Pro / Flash | Fast, context-rich                |
| **OpenAI**            | GPT-4 / GPT-4o         | General-purpose, strong reasoning |
| **Anthropic**         | Claude 3.5             | Great for summarization & QA      |
| **Mistral / Mixtral** | Open-weight            | Cost-efficient & customizable     |

---

### 🔹 Why Grounding Matters

Without retrieved evidence, LLMs may hallucinate.
Grounding ensures factual correctness and traceability (“source-aware generation”).

---

### 🔹 Summary — LLM in the RAG Loop

| Stage | Task                                       |
| ----- | ------------------------------------------ |
| 1️⃣   | Receive query + retrieved chunks           |
| 2️⃣   | Generate grounded answer                   |
| 3️⃣   | Optionally cite sources or rank evidence   |
| 4️⃣   | Return to controller for formatting/output |

---

## 🧩 6️⃣ (Bonus) Memory / Cache Layer

Some RAG systems introduce **session memory** (as you built in `agent_team_stateful_v3`) to retain conversation context.

* **Short-term memory:** session-level context for chat continuity
* **Long-term memory:** persistent vector store of past interactions

This improves follow-up queries like:

> “What about Tokyo?”
> → Recognized as “weather in Tokyo” by referencing last topic (weather).

---

## 🧩 7️⃣ End-to-End Flow Summary

| Stage | Component       | Description                     |
| ----- | --------------- | ------------------------------- |
| 1️⃣   | Controller      | Orchestrates RAG process        |
| 2️⃣   | Embedding Model | Converts query to vector        |
| 3️⃣   | Retriever       | Finds relevant chunks           |
| 4️⃣   | Vector DB       | Stores & serves embeddings      |
| 5️⃣   | LLM             | Generates grounded answer       |
| 6️⃣   | Memory          | Maintains state & context       |
| 7️⃣   | Guardrails      | Ensure policy/safety compliance |

---

## 💬 Example Real-World Explanation (Interview-Ready)

> “In a RAG system, the **Controller** orchestrates the workflow — it sends the user query to the **Retriever**, which embeds the query and searches a **Vector DB** for semantically relevant chunks. These are combined and sent to the **LLM** to generate a grounded response. The **Embedding Model** ensures semantic accuracy, while **Memory** maintains continuity and the **Guardrails** ensure compliance and factuality. This modular design allows each component to evolve independently without breaking the system.”

---

Would you like me to extend this document to include **diagrams + example LangChain and ADK code snippets** showing the same flow (Controller → Retriever → LLM)?
That version can be exported as a **PDF cheat sheet for RAG interviews**.

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

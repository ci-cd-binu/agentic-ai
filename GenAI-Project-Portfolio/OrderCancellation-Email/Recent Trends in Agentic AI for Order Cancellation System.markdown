# Recent Trends in Agentic AI for Order Cancellation System

## 1. Overview
Agentic AI refers to autonomous systems that perceive, reason, and act to achieve goals, often using LLMs as their core reasoning engine. Multi-agent systems involve collaborative agents with specialized roles, enhancing complex task execution. Frameworks like CrewAI and PhiData enable scalable, modular agentic workflows. These trends are transforming GenAI by enabling dynamic, context-aware solutions.

## 2. Application in Use Case
The order cancellation system was initially implemented as a single-agent pipeline but was enhanced with agentic and multi-agent concepts to improve flexibility and autonomy:
- **Agentic AI**: An agent orchestrated cancellation prediction, email generation, and dealer interaction, adapting to customer responses.
- **Multi-Agent System**: Introduced specialized agents for prediction, email drafting, and dealer coordination, improving modularity.
- **Frameworks**: Explored CrewAI for multi-agent orchestration and PhiData for agentic RAG, aligning with recent trends.

### Agentic Workflow
- **Single Agent**:
  - Perceived: Order data, customer profiles, cancellation predictions.
  - Reasoned: Selected email tone and content based on context.
  - Acted: Generated emails and stored drafts in Firestore.
- **Multi-Agent System**:
  - **Prediction Agent**: Ran ML models to identify high-risk orders.
  - **Email Agent**: Generated empathetic emails using Mistral and RAG.
  - **Dealer Agent**: Managed dealer dashboard interactions and email approvals.
  - **Coordinator Agent**: Orchestrated communication between agents, ensuring workflow consistency.

## 3. Technical Details
- **CrewAI Implementation**:
  - Used CrewAI for role-based multi-agent workflows.[](https://www.analyticsvidhya.com/blog/2024/07/ai-agent-frameworks/)
  - Example: Defined agents with specific tasks.
    ```python
    from crewai import Agent, Task, Crew
    prediction_agent = Agent(
        role='Prediction Analyst',
        goal='Predict order cancellations',
        backstory='Expert in ML models for risk assessment',
        tools=[VertexAITool()]
    )
    email_agent = Agent(
        role='Email Drafter',
        goal='Generate empathetic emails',
        backstory='Skilled in customer communication',
        tools=[RAGTool(), MistralTool()]
    )
    crew = Crew(
        agents=[prediction_agent, email_agent],
        tasks=[
            Task(description='Predict cancellations', agent=prediction_agent),
            Task(description='Draft emails', agent=email_agent)
        ]
    )
    crew.kickoff()
    ```
- **PhiData Implementation**:
  - Used PhiData for agentic RAG, enabling dynamic knowledge retrieval.[](https://docs.phidata.com/agents)
  - Example: Configured an agent with a vector database.
    ```python
    from phi.agent import Agent
    from phi.model.openai import OpenAIChat
    from phi.vectordb.qdrant import Qdrant
    agent = Agent(
        model=OpenAIChat(id='mistral-7b'),
        knowledge=Qdrant(collection_name='silviculture_knowledge'),
        instructions=['Retrieve relevant FAQs before drafting emails']
    )
    agent.print_response('Draft an email for a delayed order')
    ```
- **Multi-Agent Coordination**:
  - Used LangGraph for graph-based workflows, defining agent dependencies.[](https://blog.premai.io/open-source-agentic-frameworks-langgraph-vs-crewai-more/)
  - Stored agent states in Firestore for persistence.

## 4. Challenges and Mitigations
- **Challenge**: Agent coordination overhead in multi-agent systems.
  - **Mitigation**: Used CrewAI’s dynamic task allocation to streamline communication; limited agents to 3 for simplicity.
- **Challenge**: High latency in agentic RAG.
  - **Mitigation**: Cached frequent queries in Qdrant and used PhiData’s async retrieval.[](https://docs.phidata.com/agents)
- **Challenge**: Debugging agent failures.
  - **Mitigation**: Integrated CrewAI’s logging with Cloud Logging; used LangGraph’s visualization for workflow debugging.
- **Challenge**: Scalability of multi-agent systems.
  - **Mitigation**: Deployed agents on Cloud Run with auto-scaling; used lightweight models (Mistral) to reduce compute.

## 5. Interview Readiness
- **Key Points**:
  - Explain agentic AI: autonomous perception, reasoning, and action using LLMs.[](https://www.getzep.com/ai-agents/introduction-to-ai-agents)
  - Discuss multi-agent systems: role-based collaboration, task allocation, and frameworks (CrewAI, PhiData, LangGraph).[](https://blog.premai.io/open-source-agentic-frameworks-langgraph-vs-crewai-more/)
  - Highlight agentic RAG for dynamic knowledge retrieval.[](https://www.analyticsvidhya.com/blog/2024/12/agentic-rag-with-phidata/)
- **Recent Trends**:
  - Multi-agent frameworks (CrewAI, PhiData, AutoGen) for collaborative workflows.[](https://www.analyticsvidhya.com/blog/2024/07/ai-agent-frameworks/)
  - Agentic RAG with iterative reasoning and reflection.[](https://aman.ai/primers/ai/agents/)
  - Automated agent design using meta-agents (e.g., ADAS).[](https://blog.premai.io/open-source-agentic-frameworks-langgraph-vs-crewai-more/)
  - Multimodal agents handling text, images, and audio.[](https://www.analyticsvidhya.com/agenticaipioneer/)
- **Articulation Tips**:
  - Describe how agentic AI improved flexibility in the use case (e.g., adapting to customer responses).
  - Compare CrewAI (role-based, collaborative) and PhiData (data-centric, RAG-focused) for different use cases.
  - Discuss scalability and debugging strategies to show production readiness.
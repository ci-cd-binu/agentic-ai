# Custom RAGTool and MistralTool for CrewAI

This markdown file provides definitions for custom tools `RAGTool` and `MistralTool` to be used with CrewAI agents.

## 1. RAGTool Definition

```python
from crewai_tools import BaseTool

class RAGTool(BaseTool):
    name = "RAG Tool"
    description = "Retrieves relevant context from knowledge base for enhanced response."

    def _run(self, query: str) -> str:
        # Integrate with a vector database or search engine
        # Placeholder logic for retrieval
        return "Relevant knowledge base snippet for: " + query
```

## 2. MistralTool Definition

```python
from crewai_tools import BaseTool
import requests

class MistralTool(BaseTool):
    name = "Mistral LLM Tool"
    description = "Calls Mistral model to generate natural language output."

    def _run(self, prompt: str) -> str:
        headers = {"Authorization": "Bearer YOUR_API_KEY"}
        data = {
            "model": "mistral-7b-instruct",
            "messages": [{"role": "user", "content": prompt}]
        }
        response = requests.post("https://api.openrouter.ai/v1/chat/completions", headers=headers, json=data)
        return response.json()['choices'][0]['message']['content']
```

## 3. Using with CrewAI Agent

```python
from crewai import Agent

email_agent = Agent(
    role='Email Drafter',
    goal='Generate empathetic emails',
    backstory='Skilled in customer communication',
    tools=[RAGTool(), MistralTool()]
)
```

## Notes

- Replace `"YOUR_API_KEY"` with a valid API key from OpenRouter or another Mistral provider.
- Ensure required packages are installed:
  ```bash
  pip install requests
  ```
- Add error handling as needed for production use.

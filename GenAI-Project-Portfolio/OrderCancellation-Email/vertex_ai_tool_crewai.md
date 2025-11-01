# Custom VertexAITool for CrewAI

This markdown file demonstrates how to define and use a custom VertexAITool with CrewAI to call Google Cloud's Vertex AI endpoints.

## 1. Custom Tool Definition

```python
from crewai_tools import BaseTool

class VertexAITool(BaseTool):
    name = "Vertex AI Prediction Tool"
    description = "Tool to call Vertex AI endpoint for prediction"

    def _run(self, input_data: str) -> str:
        # Call your Vertex AI endpoint here using Google Cloud SDK or REST
        from google.cloud import aiplatform

        endpoint = aiplatform.Endpoint(endpoint_name="your-endpoint-name")
        response = endpoint.predict([input_data])  # Adjust for your model
        return str(response)
```

## 2. Using the Custom Tool with a CrewAI Agent

```python
from crewai import Agent

prediction_agent = Agent(
    role='Prediction Analyst',
    goal='Predict order cancellations',
    backstory='Expert in ML models for risk assessment',
    tools=[VertexAITool()]
)
```

## Notes

- Replace `"your-endpoint-name"` with the actual Vertex AI endpoint name.
- Ensure that Google Cloud credentials and required permissions are set up properly.
- Install required packages using:
  ```bash
  pip install google-cloud-aiplatform
  ```


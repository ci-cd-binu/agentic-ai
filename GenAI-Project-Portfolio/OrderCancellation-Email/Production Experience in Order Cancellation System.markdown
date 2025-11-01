# Production Experience in Order Cancellation System

## 1. Overview
Production experience in GenAI involves deploying, monitoring, and maintaining systems at scale, ensuring reliability, performance, and cost efficiency. Unlike PoCs or pilots, production systems require robust error handling, monitoring, observability, and continuous improvement to meet business SLAs (Service Level Agreements).

## 2. Application in Use Case
The order cancellation system was deployed in production on GCP, processing 1,000+ orders daily, generating empathetic emails, and supporting dealer interventions. Key production requirements included:
- **Reliability**: 99.9% uptime for batch processing and email generation.
- **Scalability**: Handle peak loads during seasonal order surges.
- **Observability**: Monitor model performance, API usage, and user feedback.
- **Cost Efficiency**: Maintain operational costs within $10,000/month.

## 3. Technical Details
- **Deployment**:
  - **Infrastructure**: Cloud Run for batch processing, Vertex AI for model inference, App Engine for dealer dashboard.
  - **CI/CD**: Used Cloud Build for automated deployments; Git for versioning models, prompts, and code.
  - **Scaling**: Cloud Run auto-scaled to 10 instances during peaks; Vertex AI endpoints used GPU autoscaling.
- **Monitoring and Observability**:
  - **Tools**: Cloud Monitoring for latency and error rates, Cloud Logging for debugging, Vertex AI Explainability for model insights.
  - **Metrics**:
    - API latency: <1s for 95% of requests.
    - Model accuracy: 85% cancellation prediction accuracy.
    - Email open rate: 50%.
  - **Alerts**: Set up notifications for API failures (>5% error rate) and model drift (>10% accuracy drop).
- **Error Handling**:
  - Retried failed API calls with exponential backoff.
  - Implemented fallback prompts for LLM failures (e.g., generic email template).
  - Logged errors to Cloud Logging for root cause analysis.
- **Continuous Improvement**:
  - **Model Retraining**: Weekly retraining of prediction models using new order data via Vertex AI Pipelines.
  - **Prompt Updates**: Monthly prompt iterations based on dealer feedback.
  - **A/B Testing**: Tested Mistral vs. GPT-3.5 emails to optimize engagement.

## 4. Challenges and Mitigations
- **Challenge**: Model drift due to changing customer behavior.
  - **Mitigation**: Monitored prediction accuracy weekly; retrained models with fresh data using MLOps pipelines.
- **Challenge**: High operational costs during peak loads.
  - **Mitigation**: Optimized batch sizes and cached embeddings; used preemptible VMs for non-critical tasks.
- **Challenge**: Dealer adoption of AI-generated emails.
  - **Mitigation**: Built an intuitive React-based dashboard (App Engine, Tailwind CSS) and provided training; achieved 80% adoption rate.
- **Challenge**: Latency spikes during batch processing.
  - **Mitigation**: Increased Cloud Run instance limits and used Pub/Sub for asynchronous task queuing.

## 5. Interview Readiness
- **Key Points**:
  - Discuss production deployment strategies: CI/CD, auto-scaling, and containerization.
  - Explain observability (monitoring, logging, alerting) and its role in maintaining SLAs.
  - Highlight MLOps practices: model retraining, drift detection, and A/B testing.[](https://www.udemy.com/course/llm-gen-ai-engineer-interview-questions-with-explanation/)
- **Recent Trends**:
  - MLOps automation with tools like Kubeflow and Vertex AI Pipelines.[](https://www.advantech.com/en/resources/case-study/leveraging-genai-and-llms-for-enhanced-decision-support-in-manufacturing)
  - Canary deployments for low-risk model updates.
  - Cost-aware AI with resource optimization (e.g., spot instances, quantization).
- **Articulation Tips**:
  - Emphasize scalability and reliability (e.g., 99.9% uptime, 1,000 orders/day).
  - Discuss how monitoring and retraining ensured long-term performance.
  - Highlight user adoption strategies to show business impact.
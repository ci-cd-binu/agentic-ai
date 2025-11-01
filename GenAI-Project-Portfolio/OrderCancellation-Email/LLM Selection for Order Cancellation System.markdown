# LLM Selection for Order Cancellation System

## 1. Overview
Large Language Model (LLM) selection is a critical decision in GenAI projects, balancing performance, cost, latency, and domain-specific requirements. Key considerations include model size, fine-tuning capabilities, inference speed, and compatibility with cloud platforms. In production environments, LLMs must handle diverse tasks (e.g., text generation, classification) while maintaining scalability and cost efficiency.

## 2. Application in Use Case
In the silviculture manufacturing order cancellation system, LLMs were used to generate empathetic email interventions to address predicted cancellation reasons. The selection process involved evaluating models for:
- **Text Generation Quality**: Ability to produce coherent, empathetic emails tailored to customer profiles and cancellation reasons.
- **Fine-Tuning Feasibility**: Support for domain-specific fine-tuning to align with silviculture terminology and tone.
- **Inference Speed**: Low latency for batch processing of email drafts.
- **Cost Efficiency**: Balancing performance with operational costs on GCP.

### Models Evaluated
- **Llama (Open-Source)**: High customizability, cost-effective for on-premises or cloud deployment, but required significant fine-tuning effort.
- **Mistral**: Lightweight, efficient for text generation, with good performance on smaller datasets.
- **GPT-3.5 (OpenAI)**: Robust out-of-the-box performance, but higher cost and limited fine-tuning flexibility compared to open-source models.

### Selection Criteria
- **Performance**: Evaluated using BLEU and ROUGE scores for email quality, with human feedback for empathy and tone.
- **Scalability**: Ability to handle batch processing of 1,000+ orders daily.
- **Integration**: Compatibility with Vertex AI for fine-tuning and deployment.
- **Cost**: Annual cost projections based on inference and fine-tuning workloads.

### Final Choice
Mistral was selected for its balance of performance, cost, and fine-tuning efficiency. It was fine-tuned on a dataset of silviculture-specific email templates and customer interaction logs, achieving a 90% satisfaction rate in dealer reviews of generated emails.

## 3. Technical Details
- **Fine-Tuning Process**:
  - Dataset: 10,000 historical emails, annotated for tone and cancellation reason.
  - Tools: Vertex AI for fine-tuning, using LoRA (Low-Rank Adaptation) to reduce compute costs.
  - Hyperparameters: Learning rate = 1e-4, batch size = 16, epochs = 5.
- **Deployment**:
  - Endpoint: Vertex AI endpoint with GPU acceleration (NVIDIA T4).
  - Inference: Batch inference via Cloud Run, processing 100 orders per minute.
- **Evaluation**:
  - Metrics: BLEU score = 0.85, ROUGE-L = 0.90, human-rated empathy score = 4.5/5.
  - A/B Testing: Mistral outperformed GPT-3.5 in empathy and cost by 20%.

## 4. Challenges and Mitigations
- **Challenge**: Overfitting on small fine-tuning datasets.
  - **Mitigation**: Used data augmentation (paraphrasing emails) and regularization techniques (dropout = 0.1).
- **Challenge**: High inference costs for large models.
  - **Mitigation**: Selected Mistral for its smaller footprint; optimized batch sizes to reduce API calls.
- **Challenge**: Ensuring domain-specific terminology.
  - **Mitigation**: Incorporated silviculture glossaries into fine-tuning data.

## 5. Interview Readiness
- **Key Points**:
  - Understand trade-offs between open-source (e.g., Llama, Mistral) and proprietary models (e.g., GPT-3.5, Claude).
  - Highlight experience with fine-tuning techniques like LoRA and QLoRA for efficiency.[](https://www.analyticsvidhya.com/blog/2024/07/ai-agent-frameworks/)
  - Discuss model evaluation metrics (BLEU, ROUGE, human feedback) and their relevance to business outcomes.
- **Recent Trends**:
  - Rise of smaller, efficient models (e.g., Mistral-7B, Phi-3) for cost-sensitive applications.
  - Multi-modal LLMs (e.g., GPT-4o, Gemini 2.0) for tasks combining text and images.[](https://aman.ai/primers/ai/agents/)
  - Parameter-efficient fine-tuning (PEFT) to reduce compute costs.[](https://www.analyticsvidhya.com/blog/2024/11/generative-ai-interview-questions/)
- **Articulation Tips**:
  - Explain why Mistral was chosen over GPT-3.5: lower cost, comparable performance post-fine-tuning, and open-source flexibility.
  - Discuss how model selection aligns with business goals (e.g., reducing cancellations by 15%).
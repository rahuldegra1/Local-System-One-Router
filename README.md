# Local System One AI Router & Code Synthesis Engine

A private, zero-recurring-cost AI automation pipeline built for deterministic decision-making and offline script generation. By executing single-forward-pass logit extraction directly on local consumer hardware, this service delivers millisecond-latency classification and routing without API token fees, network latency, or formatting hallucinations.

---

## Key Features

* **Zero-Cost Classification:** Bypasses auto-regressive text generation loops by directly sampling next-token logits for discrete options and scoring matrices.
* **Deterministic Output:** Eliminates JSON parsing errors and hallucinations through constrained mathematical logit normalization across target candidates.
* **Complete Data Privacy:** All inputs and outputs remain local to the host hardware, ensuring no proprietary data leaves the local network.
* **Greedy Code Generation Layer:** Dedicated `/v1/generate_code` endpoint formatted with ChatML system instructions to stream raw, unadorned, executable scripts directly to disk.

---

## Architectural Breakdown

### 1. System One: Deterministic Routing Engine
Rather than asking a model to produce conversational text explaining a decision, this architecture evaluates candidate token logit distributions in a single forward pass:

$$\text{Logits} \rightarrow \text{Softmax}(\text{Candidate Subset}) \rightarrow \text{Probability Distribution}$$

* `/v1/choice`: Multi-class workflow routing, support ticket prioritization, and department triage.
* `/v1/noul`: High-speed binary assertion evaluation ($P(\text{True}) \in [0.0, 1.0]$) for guardrails and rule validation.
* `/v1/score`: Continuous numeric assessment against structured rubric scales for sentiment and SLA risk indexing.

### 2. System Two: Structured Code Synthesis
Client tools request natural-language coding tasks via `/v1/generate_code`. The endpoint uses greedy decoding (`do_sample=False`, $T=0.1$) and passes through an automated syntax sanitation filter that strips redundant code fences before outputting raw text directly into destination workspace scripts.

---

## Tech Stack & Hardware Configuration

* **Model:** `Qwen/Qwen2.5-7B-Instruct`
* **VRAM Footprint:** ~5.4 GB allocated in local GPU VRAM
* **Backend:** FastAPI, Uvicorn, PyTorch, Hugging Face Transformers
* **Environment:** Python 3.12 (Isolated Virtual Environment)
* **Inference Platform:** NVIDIA GeForce RTX Laptop GPU (CUDA-accelerated)

---

## API Reference

| Endpoint | Method | Input Contract | Description |
| :--- | :--- | :--- | :--- |
| `/v1/choice` | `POST` | `{"state": str, "options": string[]}` | Returns highest-confidence selection with relative probabilities |
| `/v1/noul` | `POST` | `{"state": str, "statement": str}` | Returns binary truth confidence score |
| `/v1/score` | `POST` | `{"state": str, "rubric": dict}` | Returns weighted numeric score based on rubric mapping |
| `/v1/generate_code` | `POST` | `{"prompt": str, "language": str}` | Outputs clean, executable code strings without markdown fences |

---

## Quickstart

### 1. Clone & Setup
```bash
git clone [https://github.com/rahuldegra1/Local-System-One-Router.git](https://github.com/rahuldegra1/Local-System-One-Router.git)
cd Local-System-One-Router
python -m venv env
.\env\Scripts\activate
pip install fastapi uvicorn torch transformers numpy requests

# Coding Platform Module (Backend)

A lightweight backend module for an automated interview platform. It generates coding questions and test cases with Groq-hosted models via LangChain + LangGraph, evaluates submissions with both Judge0 execution and LLM-based scoring, and produces structured scoring feedback.

## Features
- **AI-driven generation**: questions/testcases via Groq models (`langchain-groq`) orchestrated with LangGraph.
- **Experience-based sets**: number of questions adapts to years of experience.
- **Online judge integration**: uses **Judge0 CE** for execution and correctness.
- **LLM evaluation**: LLM scores correctness/efficiency/quality with actionable feedback.
- **API-first**: clean REST endpoints ready for a future frontend.

## API overview
- `GET /health`
- `POST /ai/question` – generate a coding question
- `POST /ai/testcases` – generate test cases for a question
- `POST /ai/generate` – generate question + test cases together
- `POST /ai/questionset` – generate a question set based on years of experience
- `POST /evaluate` – evaluate a submission
- `POST /plagiarism/check` – similarity + AI-likelihood signals for submissions
- `POST /integrity/check` – suspicious imports/calls + obfuscation risk signals

## Configuration
Environment variables (optional):
- `JUDGE0_BASE_URL` (default: `https://ce.judge0.com`)
- `GROQ_API_KEY` (required for AI generation)
- `GROQ_MODEL` (default: `llama-3.1-8b-instant`)
- `AI_MAX_TOKENS` (default: `220`)
- `AI_TEMPERATURE` (default: `0.7`)

## Development
Install dependencies and run the API:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Demo
1) Export your Groq API key:

```bash
export GROQ_API_KEY="your_key_here"
```

2) Run the API:

```bash
uvicorn app.main:app --reload
```

3) In another terminal, run the demo flow:

```bash
python scripts/demo_flow.py
```

To demo a real submission, place a solution in a file and pass it via `SOLUTION_PATH`:

```bash
export SOLUTION_PATH="/path/to/solution.py"
python scripts/demo_flow.py
```

## Troubleshooting
If question generation fails with a JSON error, run the diagnostics script to isolate the failing step:

```bash
python scripts/diagnose_ai.py
```

To point at a different host/port, set:

```bash
export API_BASE_URL="http://localhost:8000"
```

## Notes
- For production, you can switch to larger Groq models and a managed Judge0 instance.
- The evaluation endpoint expects the original problem prompt in the request.

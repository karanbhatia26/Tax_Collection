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

## API reference
Below are the payloads the frontend can send and the shapes you can expect back. These examples are trimmed for readability.

### `GET /health`
**Response**

```json
{ "status": "ok" }
```

### `POST /ai/question`
Generates a single question.

**Request**

```json
{
	"role": "Backend Engineer",
	"topic": "arrays",
	"difficulty": "medium",
	"language": "python",
	"constraints": "O(n) time"
}
```

**Response**

```json
{
	"question_id": "uuid",
	"prompt": "...",
	"function_signature": "...",
	"input_format": "...",
	"output_format": "...",
	"constraints": "...",
	"examples": [{ "input": "...", "output": "...", "explanation": "..." }]
}
```

### `POST /ai/testcases`
Generates test cases for a given question prompt.

**Request**

```json
{
	"question_id": "uuid",
	"prompt": "Question text...",
	"language": "python",
	"num_cases": 5,
	"include_edge_cases": true
}
```

**Response**

```json
{
	"question_id": "uuid",
	"test_cases": [
		{ "input": "...", "expected_output": "...", "weight": 1.0, "time_limit_ms": 2000 }
	]
}
```

### `POST /ai/generate`
Generates a question and its test cases in one call.

**Request**

```json
{
	"role": "Backend Engineer",
	"topic": "arrays",
	"difficulty": "medium",
	"language": "python",
	"constraints": "...",
	"num_cases": 5,
	"include_edge_cases": true
}
```

**Response**

```json
{
	"question": { "question_id": "uuid", "prompt": "...", "function_signature": "..." },
	"test_cases": [{ "input": "...", "expected_output": "...", "weight": 1.0, "time_limit_ms": 2000 }]
}
```

### `POST /ai/questionset`
Generates a set of questions based on years of experience.

**Request**

```json
{
	"role": "Backend",
	"topic": "DSA",
	"difficulty": "medium",
	"language": "python",
	"constraints": "...",
	"years_experience": 2,
	"num_cases": 4,
	"include_edge_cases": true
}
```

**Response**

```json
{
	"num_questions": 3,
	"questions": [
		{
			"question": { "question_id": "uuid", "prompt": "..." },
			"test_cases": [{ "input": "...", "expected_output": "..." }]
		}
	]
}
```

### `POST /evaluate`
Runs the submission against Judge0 and returns LLM scoring + per‑test results.

**Request**

```json
{
	"language": "python",
	"prompt": "Question text...",
	"source_code": "print('hello')",
	"test_cases": [{ "input": "...", "expected_output": "..." }]
}
```

**Response**

```json
{
	"correctness_score": 80,
	"efficiency_score": 70,
	"quality_score": 90,
	"overall_score": 80,
	"feedback": "...",
	"results": [{ "test_case_id": 1, "status": "Accepted", "time_ms": 12.3 }]
}
```

### `POST /plagiarism/check`
Compares a candidate submission against references.

**Request**

```json
{
	"candidate_code": "def add(a,b): return a+b",
	"reference_codes": ["def add(x,y): return x+y"],
	"language": "python"
}
```

**Response**

```json
{
	"similarity_score": 76.2,
	"token_jaccard": 62.5,
	"sequence_ratio": 71.0,
	"shingle_jaccard": 68.3,
	"ai_likelihood": 18.0,
	"signals": ["high_similarity_to_reference"]
}
```

### `POST /integrity/check`
Flags risky patterns (suspicious imports, dangerous calls, obfuscation).

**Request**

```json
{ "candidate_code": "import os\nos.system('ls')", "language": "python" }
```

**Response**

```json
{ "risk_score": 64.0, "signals": ["suspicious_imports:os", "dangerous_calls:system"] }
```

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

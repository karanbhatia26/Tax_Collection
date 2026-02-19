from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class Example(BaseModel):
    input: str
    output: str
    explanation: Optional[str] = None


class QuestionRequest(BaseModel):
    role: str = Field(..., description="Job role or target position")
    topic: str = Field(..., description="Topic or skill focus")
    difficulty: str = Field("medium", description="easy | medium | hard")
    language: str = Field("python", description="Preferred language")
    constraints: Optional[str] = None


class QuestionResponse(BaseModel):
    question_id: str
    prompt: str
    function_signature: str
    input_format: str
    output_format: str
    constraints: str
    examples: List[Example]


class TestCase(BaseModel):
    __test__ = False
    input: str
    expected_output: str
    weight: float = 1.0
    time_limit_ms: int = 2000


class TestCaseRequest(BaseModel):
    question_id: str
    prompt: str
    language: str = "python"
    num_cases: int = 5
    include_edge_cases: bool = True


class TestCaseResponse(BaseModel):
    question_id: str
    test_cases: List[TestCase]


class GenerateRequest(BaseModel):
    role: str
    topic: str
    difficulty: str = "medium"
    language: str = "python"
    constraints: Optional[str] = None
    num_cases: int = 5
    include_edge_cases: bool = True


class GenerateResponse(BaseModel):
    question: QuestionResponse
    test_cases: List[TestCase]


class QuestionBundle(BaseModel):
    question: QuestionResponse
    test_cases: List[TestCase]


class QuestionSetRequest(BaseModel):
    role: str
    topic: str
    difficulty: str = "medium"
    language: str = "python"
    constraints: Optional[str] = None
    years_experience: int = Field(..., ge=0)
    num_cases: int = 5
    include_edge_cases: bool = True


class QuestionSetResponse(BaseModel):
    num_questions: int
    questions: List[QuestionBundle]


class EvaluationRequest(BaseModel):
    language: str = "python"
    language_id: Optional[int] = None
    prompt: str
    source_code: str
    test_cases: List[TestCase]


class SubmissionResult(BaseModel):
    test_case_id: int
    status: str
    stdout: Optional[str] = None
    stderr: Optional[str] = None
    time_ms: Optional[float] = None
    memory_kb: Optional[int] = None


class EvaluationResponse(BaseModel):
    correctness_score: float
    efficiency_score: float
    quality_score: float
    overall_score: float
    feedback: str
    results: List[SubmissionResult]


class PlagiarismCheckRequest(BaseModel):
    candidate_code: str
    reference_codes: List[str]
    language: Optional[str] = "python"


class PlagiarismCheckResponse(BaseModel):
    similarity_score: float
    token_jaccard: float
    sequence_ratio: float
    shingle_jaccard: float
    ai_likelihood: float
    signals: List[str]


class IntegrityCheckRequest(BaseModel):
    candidate_code: str
    language: Optional[str] = "python"


class IntegrityCheckResponse(BaseModel):
    risk_score: float
    signals: List[str]

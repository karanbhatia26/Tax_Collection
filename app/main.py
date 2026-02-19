from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException

from app.core.config import Settings, get_settings
from app.models import (
    EvaluationRequest,
    EvaluationResponse,
    GenerateRequest,
    GenerateResponse,
    IntegrityCheckRequest,
    IntegrityCheckResponse,
    PlagiarismCheckRequest,
    PlagiarismCheckResponse,
    QuestionBundle,
    QuestionRequest,
    QuestionResponse,
    QuestionSetRequest,
    QuestionSetResponse,
    TestCaseRequest,
    TestCaseResponse,
)
from app.services.ai_service import AIService
from app.services.evaluator import Evaluator
from app.services.judge0_client import Judge0Client
from app.services.integrity import integrity_check
from app.services.plagiarism import plagiarism_check

app = FastAPI(title="Coding Platform Module", version="0.1.0")


def get_ai_service(settings: Settings = Depends(get_settings)) -> AIService:
    return AIService(settings)


def get_evaluator(settings: Settings = Depends(get_settings)) -> Evaluator:
    return Evaluator(settings, Judge0Client(settings))


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ai/question", response_model=QuestionResponse)
def generate_question(request: QuestionRequest, ai_service: AIService = Depends(get_ai_service)) -> QuestionResponse:
    try:
        return ai_service.generate_question(
            role=request.role,
            topic=request.topic,
            difficulty=request.difficulty,
            language=request.language,
            constraints=request.constraints,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/ai/testcases", response_model=TestCaseResponse)
def generate_testcases(request: TestCaseRequest, ai_service: AIService = Depends(get_ai_service)) -> TestCaseResponse:
    try:
        cases = ai_service.generate_testcases(request)
        return TestCaseResponse(question_id=request.question_id, test_cases=cases)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/ai/generate", response_model=GenerateResponse)
def generate_all(request: GenerateRequest, ai_service: AIService = Depends(get_ai_service)) -> GenerateResponse:
    try:
        bundle = ai_service.generate_question_bundle(
            role=request.role,
            topic=request.topic,
            difficulty=request.difficulty,
            language=request.language,
            constraints=request.constraints,
            num_cases=request.num_cases,
            include_edge_cases=request.include_edge_cases,
        )
        return GenerateResponse(question=bundle.question, test_cases=bundle.test_cases)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/ai/questionset", response_model=QuestionSetResponse)
def generate_question_set(
    request: QuestionSetRequest,
    ai_service: AIService = Depends(get_ai_service),
) -> QuestionSetResponse:
    try:
        count = ai_service.questions_for_experience(request.years_experience)
        questions: list[QuestionBundle] = []
        for _ in range(count):
            questions.append(
                ai_service.generate_question_bundle(
                    role=request.role,
                    topic=request.topic,
                    difficulty=request.difficulty,
                    language=request.language,
                    constraints=request.constraints,
                    num_cases=request.num_cases,
                    include_edge_cases=request.include_edge_cases,
                )
            )
        return QuestionSetResponse(num_questions=count, questions=questions)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/evaluate", response_model=EvaluationResponse)
def evaluate_submission(
    request: EvaluationRequest,
    evaluator: Evaluator = Depends(get_evaluator),
) -> EvaluationResponse:
    if not request.test_cases:
        raise HTTPException(status_code=400, detail="test_cases cannot be empty")

    return evaluator.evaluate(request)


@app.post("/plagiarism/check", response_model=PlagiarismCheckResponse)
def plagiarism_endpoint(request: PlagiarismCheckRequest) -> PlagiarismCheckResponse:
    if not request.reference_codes:
        raise HTTPException(status_code=400, detail="reference_codes cannot be empty")
    try:
        result = plagiarism_check(request.candidate_code, request.reference_codes)
        return PlagiarismCheckResponse(**result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/integrity/check", response_model=IntegrityCheckResponse)
def integrity_endpoint(request: IntegrityCheckRequest) -> IntegrityCheckResponse:
    try:
        score, signals = integrity_check(request.candidate_code)
        return IntegrityCheckResponse(risk_score=score, signals=signals)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

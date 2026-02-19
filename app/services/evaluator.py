from __future__ import annotations

import json
import re
from typing import List, Tuple, TypedDict

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph

from app.core.config import Settings
from app.models import EvaluationRequest, EvaluationResponse, SubmissionResult, TestCase
from app.services.judge0_client import Judge0Client
from app.services.prompts import EVALUATION_TEMPLATE


class EvaluationState(TypedDict):
    request: EvaluationRequest
    results: List[dict]
    submission_results: List[SubmissionResult]
    llm_scores: dict


class LLMEvaluation(BaseModel):
    correctness_score: float = Field(..., ge=0, le=100)
    efficiency_score: float = Field(..., ge=0, le=100)
    quality_score: float = Field(..., ge=0, le=100)
    feedback: str
class Evaluator:
    def __init__(self, settings: Settings, judge_client: Judge0Client) -> None:
        self.settings = settings
        self.judge_client = judge_client
        self._llm = None
        self._graph = self._build_graph()

    def evaluate(self, request: EvaluationRequest) -> EvaluationResponse:
        state = self._graph.invoke({"request": request})
        scores = state["llm_scores"]
        overall = round(
            (scores["correctness_score"] * 0.6)
            + (scores["efficiency_score"] * 0.2)
            + (scores["quality_score"] * 0.2),
            2,
        )

        return EvaluationResponse(
            correctness_score=scores["correctness_score"],
            efficiency_score=scores["efficiency_score"],
            quality_score=scores["quality_score"],
            overall_score=overall,
            feedback=scores["feedback"],
            results=state["submission_results"],
        )

    def _format_results(self, results: List[dict]) -> Tuple[List[SubmissionResult], List[float]]:
        submission_results = []
        timings = []
        for idx, result in enumerate(results, start=1):
            status = result.get("status", {}).get("description", "Unknown")
            time_s = result.get("time")
            time_ms = float(time_s) * 1000 if time_s else None
            if time_ms is not None:
                timings.append(time_ms)

            submission_results.append(
                SubmissionResult(
                    test_case_id=idx,
                    status=status,
                    stdout=result.get("stdout"),
                    stderr=result.get("stderr"),
                    time_ms=time_ms,
                    memory_kb=result.get("memory"),
                )
            )
        return submission_results, timings

    def _build_graph(self):
        graph = StateGraph(EvaluationState)
        graph.add_node("run_judge", self._run_judge_node)
        graph.add_node("llm_score", self._llm_score_node)
        graph.set_entry_point("run_judge")
        graph.add_edge("run_judge", "llm_score")
        graph.add_edge("llm_score", END)
        return graph.compile()

    def _run_judge_node(self, state: EvaluationState) -> EvaluationState:
        request = state["request"]
        language_id = request.language_id or self.settings.language_map.get(
            request.language.lower(), self.settings.default_language_id
        )
        results = self.judge_client.run_batch(
            language_id=language_id,
            source_code=request.source_code,
            testcases=[case.model_dump() for case in request.test_cases],
        )
        submission_results, _ = self._format_results(results)
        state["results"] = results
        state["submission_results"] = submission_results
        return state

    def _llm_score_node(self, state: EvaluationState) -> EvaluationState:
        request = state["request"]
        results_json = json.dumps(state["results"], ensure_ascii=False)
        parser = PydanticOutputParser(pydantic_object=LLMEvaluation)
        prompt = EVALUATION_TEMPLATE.format(
            language=request.language,
            prompt=request.prompt,
            source_code=request.source_code,
            results_json=results_json,
        )
        formatted_prompt = f"{prompt}\n\n{parser.get_format_instructions()}"
        response = self._get_llm().invoke([HumanMessage(content=formatted_prompt)])
        try:
            scores = parser.parse(response.content).model_dump()
        except Exception:
            scores = self._extract_json(response.content)
        state["llm_scores"] = self._normalize_scores(scores)
        return state

    def _get_llm(self) -> ChatGroq:
        if self._llm is None:
            if not self.settings.groq_api_key:
                raise RuntimeError("GROQ_API_KEY is not configured")
            self._llm = ChatGroq(
                api_key=self.settings.groq_api_key,
                model=self.settings.groq_model,
                temperature=self.settings.ai_temperature,
                max_tokens=self.settings.ai_max_tokens,
            )
        return self._llm

    def _extract_json(self, generated: str) -> dict:
        cleaned = self._strip_code_fences(generated)
        candidate = self._find_json_segment(cleaned)
        if candidate is None:
            return self._repair_json(generated)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return self._repair_json(candidate)

    def _repair_json(self, payload: str) -> dict:
        repair_prompt = (
            "Fix the following text into valid JSON only. "
            "Return a JSON object and nothing else.\n\n"
            f"Text:\n{payload}"
        )
        response = self._get_llm().invoke([HumanMessage(content=repair_prompt)])
        cleaned = self._strip_code_fences(response.content)
        candidate = self._find_json_segment(cleaned)
        if candidate is None:
            raise ValueError("LLM evaluation did not return JSON")
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            raise ValueError("Failed to parse LLM evaluation JSON") from exc

    def _strip_code_fences(self, text: str) -> str:
        text = text.strip()
        text = re.sub(r"^```(json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()
        return text

    def _find_json_segment(self, text: str) -> str | None:
        start = text.find("{")
        if start == -1:
            return None

        depth = 0
        in_string = False
        escape = False
        for idx in range(start, len(text)):
            char = text[idx]
            if escape:
                escape = False
                continue
            if char == "\\":
                escape = True
                continue
            if char == '"':
                in_string = not in_string
            if in_string:
                continue
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start : idx + 1]
        return None

    def _normalize_scores(self, payload: dict) -> dict:
        correctness = float(payload.get("correctness_score", 0.0))
        efficiency = float(payload.get("efficiency_score", 0.0))
        quality = float(payload.get("quality_score", 0.0))
        feedback = str(payload.get("feedback", "No feedback provided."))
        return {
            "correctness_score": max(0.0, min(correctness, 100.0)),
            "efficiency_score": max(0.0, min(efficiency, 100.0)),
            "quality_score": max(0.0, min(quality, 100.0)),
            "feedback": feedback,
        }

from __future__ import annotations

import json
import logging
import random
import re
import uuid
from typing import List, TypedDict

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from app.core.config import Settings
from app.models import Example, QuestionBundle, QuestionResponse, TestCase, TestCaseRequest
from app.services.prompts import QUESTION_TEMPLATE, TESTCASE_TEMPLATE


class QuestionState(TypedDict):
    role: str
    topic: str
    difficulty: str
    language: str
    constraints: str | None
    num_cases: int
    include_edge_cases: bool
    question: dict
    testcases: List[dict]


class LLMExample(BaseModel):
    input: str
    output: str
    explanation: str | None = None


class LLMQuestion(BaseModel):
    prompt: str
    function_signature: str
    input_format: str
    output_format: str
    constraints: str
    examples: List[LLMExample] = Field(default_factory=list)


class LLMTestCase(BaseModel):
    input: str
    expected_output: str
    weight: float = 1.0
    time_limit_ms: int = 2000


class LLMTestCaseSet(BaseModel):
    test_cases: List[LLMTestCase]


class AIService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._llm = None
        self._graph = self._build_graph()
        self._logger = logging.getLogger(__name__)

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

    def _generate_text(self, prompt: str) -> str:
        llm = self._get_llm()
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content.strip()

    def _build_graph(self):
        graph = StateGraph(QuestionState)
        graph.add_node("generate_question", self._generate_question_node)
        graph.add_node("generate_testcases", self._generate_testcases_node)
        graph.set_entry_point("generate_question")
        graph.add_edge("generate_question", "generate_testcases")
        graph.add_edge("generate_testcases", END)
        return graph.compile()

    def _generate_question_node(self, state: QuestionState) -> QuestionState:
        prompt = QUESTION_TEMPLATE.format(
            role=state["role"],
            topic=state["topic"],
            difficulty=state["difficulty"],
            language=state["language"],
            constraints=state.get("constraints") or "None",
        )
        try:
            question = self._generate_question_with_retries(prompt)
        except Exception as exc:
            self._logger.exception("Question generation failed")
            question = self._fallback_question(state)
        state["question"] = self._ensure_question_schema(question)
        return state

    def _generate_testcases_node(self, state: QuestionState) -> QuestionState:
        prompt = TESTCASE_TEMPLATE.format(
            prompt=state["question"].get("prompt", ""),
            language=state["language"],
            num_cases=state["num_cases"],
            include_edge_cases=state["include_edge_cases"],
        )
        try:
            testcases = self._generate_testcases_with_retries(prompt)
            testcases = self._ensure_testcase_schema(testcases)
        except Exception as exc:
            self._logger.exception("Testcase generation failed")
            testcases = self._build_testcases_from_examples(state.get("question", {}), state["num_cases"])
            if not testcases:
                testcases = self._fallback_testcases(state["num_cases"])
        state["testcases"] = testcases
        return state

    def generate_question(self, role: str, topic: str, difficulty: str, language: str, constraints: str | None) -> QuestionResponse:
        state = self._graph.invoke(
            {
                "role": role,
                "topic": topic,
                "difficulty": difficulty,
                "language": language,
                "constraints": constraints,
                "num_cases": 1,
                "include_edge_cases": True,
            }
        )
        question = self._normalize_question(state["question"])
        question_id = str(uuid.uuid4())
        return QuestionResponse(question_id=question_id, **question)

    def generate_testcases(self, request: TestCaseRequest) -> List[TestCase]:
        prompt = TESTCASE_TEMPLATE.format(
            prompt=request.prompt,
            language=request.language,
            num_cases=request.num_cases,
            include_edge_cases=request.include_edge_cases,
        )
        extracted = self._generate_testcases_with_retries(prompt)
        extracted = self._ensure_testcase_schema(extracted)
        cases = self._normalize_testcases(extracted, request.num_cases)
        random.shuffle(cases)
        return cases[: request.num_cases]

    def generate_question_bundle(
        self,
        role: str,
        topic: str,
        difficulty: str,
        language: str,
        constraints: str | None,
        num_cases: int,
        include_edge_cases: bool,
    ) -> QuestionBundle:
        state = self._graph.invoke(
            {
                "role": role,
                "topic": topic,
                "difficulty": difficulty,
                "language": language,
                "constraints": constraints,
                "num_cases": num_cases,
                "include_edge_cases": include_edge_cases,
            }
        )
        question = self._normalize_question(self._ensure_question_schema(state["question"]))
        question_id = str(uuid.uuid4())
        testcases = self._normalize_testcases(self._ensure_testcase_schema(state["testcases"]), num_cases)
        return QuestionBundle(
            question=QuestionResponse(question_id=question_id, **question),
            test_cases=testcases,
        )

    def questions_for_experience(self, years_experience: int) -> int:
        if years_experience < 2:
            return 2
        if years_experience < 4:
            return 3
        if years_experience < 7:
            return 4
        return 5

    def _extract_json(self, generated: str, *, expect_list: bool) -> list | dict:
        cleaned = self._strip_code_fences(generated)
        candidate = self._find_json_segment(cleaned, expect_list=expect_list)
        if candidate is None:
            return self._repair_json(generated, expect_list=expect_list)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return self._repair_json(candidate, expect_list=expect_list)

    def _repair_json(self, payload: str, *, expect_list: bool) -> list | dict:
        repair_prompt = (
            "Fix the following text into valid JSON only. "
            f"Return a {'JSON array' if expect_list else 'JSON object'} and nothing else.\n\n"
            f"Text:\n{payload}"
        )
        repaired = self._generate_text(repair_prompt)
        cleaned = self._strip_code_fences(repaired)
        candidate = self._find_json_segment(cleaned, expect_list=expect_list)
        if candidate is None:
            raise ValueError("AI response did not contain valid JSON")
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            raise ValueError("Failed to parse AI JSON output") from exc

    def _strip_code_fences(self, text: str) -> str:
        text = text.strip()
        text = re.sub(r"^```(json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()
        return text

    def _find_json_segment(self, text: str, *, expect_list: bool) -> str | None:
        opener = "[" if expect_list else "{"
        closer = "]" if expect_list else "}"
        start = text.find(opener)
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
            if char == opener:
                depth += 1
            elif char == closer:
                depth -= 1
                if depth == 0:
                    return text[start : idx + 1]
        return None

    def _normalize_question(self, payload: dict) -> dict:
        examples = []
        for example in payload.get("examples", []):
            examples.append(
                Example(
                    input=str(example.get("input", "")),
                    output=str(example.get("output", "")),
                    explanation=(
                        str(example.get("explanation")) if example.get("explanation") is not None else None
                    ),
                )
            )
        return {
            "prompt": payload["prompt"],
            "function_signature": payload["function_signature"],
            "input_format": payload["input_format"],
            "output_format": payload["output_format"],
            "constraints": payload["constraints"],
            "examples": examples,
        }

    def _invoke_question_parser(self, prompt: str) -> dict:
        parser = PydanticOutputParser(pydantic_object=LLMQuestion)
        formatted_prompt = f"{prompt}\n\n{parser.get_format_instructions()}"
        response = self._get_llm().invoke([HumanMessage(content=formatted_prompt)])
        parsed = parser.parse(response.content)
        return parsed.model_dump()

    def _invoke_testcase_parser(self, prompt: str) -> list:
        parser = PydanticOutputParser(pydantic_object=LLMTestCaseSet)
        formatted_prompt = f"{prompt}\n\n{parser.get_format_instructions()}"
        response = self._get_llm().invoke([HumanMessage(content=formatted_prompt)])
        parsed = parser.parse(response.content)
        payload = parsed.model_dump()
        return payload.get("test_cases", payload)

    def _generate_question_with_retries(self, prompt: str) -> dict:
        try:
            return self._invoke_question_parser(prompt)
        except Exception:
            strict_prompt = (
                "Return only valid JSON. Do not add explanations.\n\n" + prompt
            )
            generated = self._generate_text(strict_prompt)
            return self._extract_json(generated, expect_list=False)

    def _generate_testcases_with_retries(self, prompt: str) -> list:
        try:
            return self._invoke_testcase_parser(prompt)
        except Exception:
            strict_prompt = (
                "Return only valid JSON. Do not add explanations.\n\n" + prompt
            )
            generated = self._generate_text(strict_prompt)
            return self._extract_json(generated, expect_list=True)

    def _ensure_question_schema(self, payload: dict) -> dict:
        required = {"prompt", "function_signature", "input_format", "output_format", "constraints", "examples"}
        if required.issubset(payload.keys()):
            return payload

        repair_prompt = (
            "Convert the following data into valid JSON with this schema only:\n"
            "{\n"
            "  \"prompt\": \"...\",\n"
            "  \"function_signature\": \"...\",\n"
            "  \"input_format\": \"...\",\n"
            "  \"output_format\": \"...\",\n"
            "  \"constraints\": \"...\",\n"
            "  \"examples\": [{\"input\": \"...\", \"output\": \"...\", \"explanation\": \"...\"}]\n"
            "}\n\n"
            f"Data:\n{json.dumps(payload, ensure_ascii=False)}"
        )
        repaired = self._generate_text(repair_prompt)
        repaired_json = self._extract_json(repaired, expect_list=False)
        if not required.issubset(repaired_json.keys()):
            missing = required - repaired_json.keys()
            raise ValueError(f"Question JSON missing required fields: {sorted(missing)}")
        return repaired_json

    def _ensure_testcase_schema(self, payload: list) -> list:
        cleaned = self._normalize_testcase_payload(payload)
        if cleaned:
            return cleaned

        repair_prompt = (
            "Convert the following data into valid JSON array with schema:\n"
            "[{\"input\": \"...\", \"expected_output\": \"...\", \"weight\": 1.0, \"time_limit_ms\": 2000}]\n\n"
            f"Data:\n{json.dumps(payload, ensure_ascii=False)}"
        )
        repaired = self._generate_text(repair_prompt)
        repaired_json = self._extract_json(repaired, expect_list=True)
        cleaned = self._normalize_testcase_payload(repaired_json)
        if not cleaned:
            raise ValueError("Test case JSON missing required fields")
        return cleaned

    def _normalize_testcase_payload(self, payload: object) -> list:
        if isinstance(payload, dict) and "test_cases" in payload:
            payload = payload["test_cases"]

        if not isinstance(payload, list):
            return []

        cleaned = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            input_value = item.get("input") or item.get("stdin") or item.get("in")
            expected = item.get("expected_output") or item.get("output") or item.get("expected")
            if input_value is None or expected is None:
                continue
            cleaned.append(
                {
                    "input": input_value,
                    "expected_output": expected,
                    "weight": item.get("weight", 1.0),
                    "time_limit_ms": item.get("time_limit_ms", 2000),
                }
            )
        return cleaned

    def _build_testcases_from_examples(self, question: dict, num_cases: int) -> list:
        examples = question.get("examples", []) if isinstance(question, dict) else []
        cases = []
        for example in examples:
            if not isinstance(example, dict):
                continue
            input_value = example.get("input")
            expected = example.get("output")
            if input_value is None or expected is None:
                continue
            cases.append(
                {
                    "input": input_value,
                    "expected_output": expected,
                    "weight": 1.0,
                    "time_limit_ms": 2000,
                }
            )
        return cases[:num_cases]

    def _fallback_question(self, state: QuestionState) -> dict:
        topic = state.get("topic", "arrays")
        language = state.get("language", "python")
        return {
            "prompt": f"Write a {language} function that solves a {topic} problem described by the inputs.",
            "function_signature": "def solve(input_data):",
            "input_format": "Input data as described in the prompt.",
            "output_format": "Output as described in the prompt.",
            "constraints": state.get("constraints") or "None",
            "examples": [],
        }

    def _fallback_testcases(self, num_cases: int) -> list:
        return [
            {"input": "[]", "expected_output": "[]", "weight": 1.0, "time_limit_ms": 2000}
            for _ in range(max(1, num_cases))
        ]

    def _normalize_testcases(self, payload: list, num_cases: int) -> List[TestCase]:
        cases = []
        for item in payload[:num_cases]:
            cases.append(
                TestCase(
                    input=str(item.get("input", "")),
                    expected_output=str(item.get("expected_output", "")),
                    weight=float(item.get("weight", 1.0)),
                    time_limit_ms=int(item.get("time_limit_ms", 2000)),
                )
            )
        if not cases:
            raise ValueError("No valid test cases produced")
        return cases

from __future__ import annotations
from typing import Dict, List
import requests
from app.core.config import Settings
class Judge0Client:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
    def run_submission(self, *, language_id: int, source_code: str, stdin: str, expected_output: str, time_limit_ms: int) -> Dict:
        payload = {
            "language_id": language_id,
            "source_code": source_code,
            "stdin": stdin,
            "expected_output": expected_output,
            "cpu_time_limit": max(time_limit_ms / 1000, 1),
        }
        response = requests.post(
            f"{self.settings.judge0_base_url}/submissions?base64_encoded=false&wait=true",
            json=payload,
            timeout=self.settings.judge0_timeout_seconds,
        )
        response.raise_for_status()
        return response.json()
    def run_batch(self, *, language_id: int, source_code: str, testcases: List[Dict]) -> List[Dict]:
        results = []
        for case in testcases:
            results.append(
                self.run_submission(
                    language_id=language_id,
                    source_code=source_code,
                    stdin=case["input"],
                    expected_output=case["expected_output"],
                    time_limit_ms=case.get("time_limit_ms", 2000),
                )
            )
        return results

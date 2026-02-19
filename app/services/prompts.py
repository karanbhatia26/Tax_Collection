# QUESTION_TEMPLATE = """
# You are an interview question generator.
# Role: {role}
# Topic: {topic}
# Difficulty: {difficulty}
# Language: {language}
# Constraints: {constraints}
# Return ONLY valid JSON with this schema:
# {{
#     "prompt": "...",
#     "function_signature": "...",
#     "input_format": "...",
#     "output_format": "...",
#     "constraints": "...",
#     "examples": [{{"input": "...", "output": "...", "explanation": "..."}}]
# }}
# """

QUESTION_TEMPLATE = """
Generate one coding question.
Role: {role}; Topic: {topic}; Difficulty: {difficulty}; Language: {language}; Constraints: {constraints}
Return ONLY JSON:
{{"prompt":"...","function_signature":"...","input_format":"...","output_format":"...","constraints":"...","examples":[{{"input":"...","output":"...","explanation":"..."}}]}}
"""

# Previous verbose prompt (kept for reference):
# TESTCASE_TEMPLATE = """
# You are a test case generator for coding challenges.
# Prompt: {prompt}
# Language: {language}
# Generate {num_cases} test cases. Include edge cases: {include_edge_cases}.
# Return ONLY valid JSON as a list with schema:
# [
#     {{"input": "...", "expected_output": "...", "weight": 1.0, "time_limit_ms": 2000}}
# ]
# """

TESTCASE_TEMPLATE = """
Generate {num_cases} test cases for this prompt. Edge cases: {include_edge_cases}. Language: {language}
Prompt: {prompt}
Return ONLY JSON list:
[{"input":"...","expected_output":"...","weight":1.0,"time_limit_ms":2000}]
"""

# Previous verbose prompt (kept for reference):
# EVALUATION_TEMPLATE = """
# You are a senior code evaluator. Score the candidate solution from 0-100.
# Language: {language}
# Problem Prompt: {prompt}
# Source Code:
# {source_code}
#
# Execution Results (JSON):
# {results_json}
#
# Return ONLY valid JSON with this schema:
# {{
#     "correctness_score": 0-100,
#     "efficiency_score": 0-100,
#     "quality_score": 0-100,
#     "feedback": "short actionable feedback"
# }}
# """

EVALUATION_TEMPLATE = """
Score solution 0-100 using prompt, code, and run results.
Lang: {language}
Prompt: {prompt}
Code: {source_code}
Results: {results_json}
Return ONLY JSON:
{{"correctness_score":0-100,"efficiency_score":0-100,"quality_score":0-100,"feedback":"..."}}
"""

from __future__ import annotations

import math
import re
from typing import List, Tuple

SUSPICIOUS_IMPORTS = {
    "os",
    "sys",
    "subprocess",
    "socket",
    "requests",
    "http",
    "urllib",
    "ftplib",
    "pickle",
    "base64",
    "marshal",
    "ctypes",
}

DANGEROUS_CALLS = {
    "eval",
    "exec",
    "compile",
    "open",
    "__import__",
    "getattr",
    "setattr",
    "globals",
    "locals",
    "input",
    "system",
    "popen",
}


def _shannon_entropy(text: str) -> float:
    if not text:
        return 0.0
    freq = {}
    for ch in text:
        freq[ch] = freq.get(ch, 0) + 1
    entropy = 0.0
    for count in freq.values():
        p = count / len(text)
        entropy -= p * math.log2(p)
    return entropy
def _extract_imports(code: str) -> List[str]:
    return re.findall(r"^\s*(?:import|from)\s+([a-zA-Z_][a-zA-Z0-9_]*)", code, re.MULTILINE)

def _extract_calls(code: str) -> List[str]:
    return re.findall(r"\b([A-Za-z_][A-Za-z_0-9]*)\s*\(", code)


def integrity_check(code: str) -> Tuple[float, List[str]]:
    signals: List[str] = []
    lines = code.splitlines()
    if not lines:
        return 0.0, ["empty_code"]

    imports = set(_extract_imports(code))
    bad_imports = sorted(imports & SUSPICIOUS_IMPORTS)
    if bad_imports:
        signals.append(f"suspicious_imports:{','.join(bad_imports)}")

    calls = set(_extract_calls(code))
    bad_calls = sorted(calls & DANGEROUS_CALLS)
    if bad_calls:
        signals.append(f"dangerous_calls:{','.join(bad_calls)}")

    long_lines = sum(1 for line in lines if len(line) > 160)
    if long_lines:
        signals.append("long_lines")

    avg_len = sum(len(line) for line in lines) / max(len(lines), 1)
    if avg_len > 120:
        signals.append("high_avg_line_length")

    entropy = _shannon_entropy(code)
    if entropy > 4.6:
        signals.append("high_entropy_obfuscation")

    non_ascii = sum(1 for ch in code if ord(ch) > 127)
    if non_ascii > 0:
        signals.append("non_ascii_chars")

    score = 0.0
    score += min(len(bad_imports) * 12, 40)
    score += min(len(bad_calls) * 12, 40)
    if long_lines:
        score += 10
    if avg_len > 120:
        score += 10
    if entropy > 4.6:
        score += 10
    if non_ascii > 0:
        score += 10

    return min(score, 100.0), signals

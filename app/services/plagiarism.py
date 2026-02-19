from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Iterable, List, Tuple


def normalize_code(code: str) -> str:
    code = re.sub(r"'''[\s\S]*?'''", "", code)
    code = re.sub(r'"""[\s\S]*?"""', "", code)
    code = re.sub(r"#.*", "", code)
    code = re.sub(r"\b\d+(\.\d+)?\b", "NUM", code)
    code = re.sub(r"(['\"][^'\"]*['\"])", "STR", code)
    code = re.sub(r"\s+", " ", code).strip()
    return code


def tokenize(code: str) -> List[str]:
    return re.findall(r"[A-Za-z_][A-Za-z_0-9]*|NUM|STR", code)


def shingle(tokens: List[str], size: int = 5) -> List[Tuple[str, ...]]:
    if len(tokens) < size:
        return [tuple(tokens)] if tokens else []
    return [tuple(tokens[i : i + size]) for i in range(len(tokens) - size + 1)]


def jaccard(a: Iterable, b: Iterable) -> float:
    set_a, set_b = set(a), set(b)
    if not set_a and not set_b:
        return 0.0
    return len(set_a & set_b) / max(len(set_a | set_b), 1)


def sequence_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def ai_likelihood_score(code: str, tokens: List[str]) -> Tuple[float, List[str]]:
    signals = []
    if not tokens:
        return 0.0, signals

    unique_ratio = len(set(tokens)) / max(len(tokens), 1)
    if unique_ratio < 0.35:
        signals.append("low_unique_token_ratio")

    line_count = max(len(code.splitlines()), 1)
    comment_ratio = code.count("#") / line_count
    if comment_ratio < 0.02:
        signals.append("low_comment_ratio")

    base = (1 - unique_ratio) * 70
    if comment_ratio < 0.02:
        base += 10
    score = min(max(base, 0.0), 100.0)
    return score, signals


def compare_codes(candidate: str, reference: str) -> Tuple[float, float, float, float]:
    cand_norm = normalize_code(candidate)
    ref_norm = normalize_code(reference)
    cand_tokens = tokenize(cand_norm)
    ref_tokens = tokenize(ref_norm)

    token_jaccard = jaccard(cand_tokens, ref_tokens)
    shingle_jaccard = jaccard(shingle(cand_tokens), shingle(ref_tokens))
    seq_ratio = sequence_ratio(cand_norm, ref_norm)

    similarity = (0.4 * shingle_jaccard + 0.4 * seq_ratio + 0.2 * token_jaccard) * 100
    return similarity, token_jaccard * 100, seq_ratio * 100, shingle_jaccard * 100


def plagiarism_check(candidate: str, references: List[str]) -> dict:
    best = {
        "similarity_score": 0.0,
        "token_jaccard": 0.0,
        "sequence_ratio": 0.0,
        "shingle_jaccard": 0.0,
    }
    for ref in references:
        similarity, token_j, seq_r, shingle_j = compare_codes(candidate, ref)
        if similarity > best["similarity_score"]:
            best = {
                "similarity_score": similarity,
                "token_jaccard": token_j,
                "sequence_ratio": seq_r,
                "shingle_jaccard": shingle_j,
            }

    norm_candidate = normalize_code(candidate)
    tokens = tokenize(norm_candidate)
    ai_score, signals = ai_likelihood_score(candidate, tokens)

    if best["similarity_score"] > 70:
        signals.append("high_similarity_to_reference")
    if best["shingle_jaccard"] > 60:
        signals.append("high_shingle_overlap")
    if best["sequence_ratio"] > 65:
        signals.append("high_sequence_similarity")

    return {
        **best,
        "ai_likelihood": ai_score,
        "signals": signals,
    }

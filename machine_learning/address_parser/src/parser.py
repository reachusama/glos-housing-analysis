"""
ONS Address Index - Probabilistic Parser

This module exposes parsing helpers for a CRFsuite address model.
- No embedded tests or prints.
- No hardcoded model path: pass it into each call.
- Efficient: taggers are cached per-model to avoid repeated opens.

Requirements
------------
pycrfsuite (https://python-crfsuite.readthedocs.io/en/latest/)

Author: Usama Shahid
Version: 0.5 (24-Aug-2025)
"""
from __future__ import annotations

from functools import lru_cache
from typing import List, Tuple, Dict, Any
import pycrfsuite

# You can still use the same tokens module by default, but callers may override.
import machine_learning.address_parser.src.tokens as default_tok


# ---- tagger management ----
@lru_cache(maxsize=8)
def _get_tagger(model_path: str) -> pycrfsuite.Tagger:
    """
    Lazily create and cache a pycrfsuite.Tagger for the given model path.
    Raises an exception if the model cannot be opened.
    """
    tagger = pycrfsuite.Tagger()
    tagger.open(model_path)  # pycrfsuite throws if the path is invalid/corrupt
    return tagger


# ---- internals ----
def _parse(raw: str, model_path: str, tok=default_tok) -> Tuple[List[str], List[str]]:
    """
    Return (tokens, tags). If input yields no tokens, return ([], []).
    """
    tokens: List[str] = tok.tokenize(raw)
    if not tokens:
        return [], []

    features = tok.tokens2features(tokens)
    if not features:
        return tokens, []

    tagger = _get_tagger(model_path)
    tags: List[str] = tagger.tag(features)
    return tokens, tags


# ---- public API ----
def parse(raw: str, model_path: str, tok=default_tok) -> List[Tuple[str, str]]:
    """
    Return a list of (token, label) pairs. Empty list for empty/untaggable input.

    Args:
        raw: Free-text address string.
        model_path: Filesystem path to the .crfsuite model file.
        tok: Tokenization/feature module (must provide tokenize, tokens2features).

    Example:
        parse("FLAT 2 10 QUEEN STREET BURY BL8 1JG", "/models/addressCRF.crfsuite")
    """
    tokens, tags = _parse(raw, model_path, tok)
    if not tokens or not tags:
        return []
    return list(zip(tokens, tags))


def parse_with_marginal_probability(
        raw: str, model_path: str, tok=default_tok
) -> List[Tuple[str, str, float]]:
    """
    Return a list of (token, label, marginal_prob). Empty list if untaggable.
    """
    tokens, tags = _parse(raw, model_path, tok)
    if not tokens or not tags:
        return []
    tagger = _get_tagger(model_path)
    marginals = [tagger.marginal(tag, i) for i, tag in enumerate(tags)]
    return list(zip(tokens, tags, marginals))


def parse_with_probabilities(
        raw: str, model_path: str, tok=default_tok
) -> Dict[str, Any]:
    """
    Return a dict with:
      - tokens: List[str]
      - tags: List[str]
      - marginal_probabilities: List[float]
      - sequence_probability: float

    If untaggable, returns empty arrays and probability 0.0.
    """
    tokens, tags = _parse(raw, model_path, tok)
    if not tokens or not tags:
        return {
            "tokens": [],
            "tags": [],
            "marginal_probabilities": [],
            "sequence_probability": 0.0,
        }

    tagger = _get_tagger(model_path)
    marginals = [tagger.marginal(tag, i) for i, tag in enumerate(tags)]
    seq_p = tagger.probability(tags)
    return {
        "tokens": tokens,
        "tags": tags,
        "marginal_probabilities": marginals,
        "sequence_probability": seq_p,
    }


def tag(raw: str, model_path: str, tok=default_tok) -> Dict[str, str]:
    """
    Return a dict: label -> 'joined tokens'. Empty dict if untaggable.
    Trailing punctuation/commas/semicolons are stripped from each component.
    """
    out: Dict[str, List[str]] = {}
    for token, label in parse(raw, model_path, tok):
        out.setdefault(label, []).append(token)

    # Join and tidy
    return {label: " ".join(parts).strip(" ,;") for label, parts in out.items()}

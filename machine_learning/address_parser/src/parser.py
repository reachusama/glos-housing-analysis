"""
ONS Address Index - Probabilistic Parser
========================================

This file defines the calling mechanism for a trained probabilistic parser model.
It also implements a simple test. Note that the results are model dependent, so
the assertions will fail if a new model is trained.


Requirements
------------

:requires: pycrfsuite (https://python-crfsuite.readthedocs.io/en/latest/)


Author
------

:author: Usama Shahid


Version
-------

:version: 0.4
:date: 22-Aug-2025
"""
# parser.py (safe version)
import os
import sys
from collections import OrderedDict

import machine_learning.address_parser.src.tokens as tok
import pycrfsuite

# ---- paths ----
MODEL_FILE = 'addressCRF.crfsuite'
# Resolve relative to this file so imports from notebooks work
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.abspath(os.path.join(_THIS_DIR, '../configs/model/training'))
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)

# ---- load tagger ----
try:
    TAGGER = pycrfsuite.Tagger()
    TAGGER.open(MODEL_PATH)
    print('Using model from', MODEL_PATH)
except Exception as e:
    print(f'ERROR: cannot open CRF model at {MODEL_PATH}\n{e}')
    sys.exit(-9)


# ---- internals ----
def _parse(raw_string):
    """
    Return (tokens, tags). If the input yields no tokens, return ([], []).
    """
    tokens = tok.tokenize(raw_string)
    if not tokens:
        return [], []

    features = tok.tokens2features(tokens)
    # pycrfsuite requires non-empty features; guard just in case
    if not features:
        return tokens, []

    tags = TAGGER.tag(features)
    return tokens, tags


# ---- public API ----
def parse(raw_string):
    """
    Return a list of (token, label) pairs. Empty list for empty/untaggable input.
    """
    tokens, tags = _parse(raw_string)
    if not tokens or not tags:
        return []
    return list(zip(tokens, tags))


def parse_with_marginal_probability(raw_string):
    """
    Return a list of (token, label, marginal_prob). Empty list if untaggable.
    """
    tokens, tags = _parse(raw_string)
    if not tokens or not tags:
        return []
    marginals = [TAGGER.marginal(tag, i) for i, tag in enumerate(tags)]
    return list(zip(tokens, tags, marginals))


def parse_with_probabilities(raw_string):
    """
    Return an OrderedDict with tokens, tags, marginal_probabilities, sequence_probability.
    If untaggable, returns empty arrays and probability 0.0.
    """
    tokens, tags = _parse(raw_string)
    if not tokens or not tags:
        return OrderedDict(tokens=[],
                           tags=[],
                           marginal_probabilites=[],
                           sequence_probability=0.0)
    marginals = [TAGGER.marginal(tag, i) for i, tag in enumerate(tags)]
    seq_p = TAGGER.probability(tags)
    return OrderedDict(tokens=tokens,
                       tags=tags,
                       marginal_probabilites=marginals,
                       sequence_probability=seq_p)


def tag(raw_string):
    """
    Return an OrderedDict label -> 'joined tokens'. Empty dict if untaggable.
    """
    out = OrderedDict()
    for token, label in parse(raw_string):
        out.setdefault(label, []).append(token)
    for label in list(out.keys()):
        component = ' '.join(out[label]).strip(' ,;')
        out[label] = component
    return out


# Optional quick check
if __name__ == "__main__":
    print(tag("FLAT 2 10 QUEEN STREET BURY BL8 1JG"))

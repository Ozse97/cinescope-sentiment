"""
text_utils.py — shared text preprocessing
==========================================
Used by BOTH model.py (training) and app.py (inference), so predictions
always see text cleaned exactly the same way it was trained on.

IMPORTANT: this function is passed to TfidfVectorizer(preprocessor=...) and
gets pickled by reference (module path + function name). Keep this file next
to app.py in deployment so unpickling can resolve it.
"""

import re

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_NON_ALPHA_RE = re.compile(r"[^a-z\s']")
_MULTI_SPACE_RE = re.compile(r"\s+")

# Ordered so longer/more specific patterns are replaced before generic ones.
_CONTRACTIONS = [
    ("won't", "will not"),
    ("can't", "cannot"),
    ("shan't", "shall not"),
    ("n't", " not"),      # isn't, don't, wasn't, wouldn't, couldn't, ...
    ("'re", " are"),
    ("'s", " is"),
    ("'d", " would"),
    ("'ll", " will"),
    ("'ve", " have"),
    ("'m", " am"),
]


def clean_text(text: str) -> str:
    """Lowercase, strip HTML, expand contractions, drop stray punctuation.

    Keeping contraction expansion is important for sentiment: 'not' is one
    of the strongest polarity signals in movie reviews, and the old
    token pattern silently threw it away (don't -> 'don' + 't').
    """
    if not isinstance(text, str):
        text = str(text)

    text = text.lower()
    text = _HTML_TAG_RE.sub(" ", text)          # <br />, <i>, etc.

    for pattern, replacement in _CONTRACTIONS:
        text = text.replace(pattern, replacement)

    text = _NON_ALPHA_RE.sub(" ", text)          # drop leftover punctuation/digits
    text = _MULTI_SPACE_RE.sub(" ", text).strip()
    return text

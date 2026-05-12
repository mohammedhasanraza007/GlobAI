"""
core/intent_classifier.py
--------------------------
Deterministic rule-based intent classifier for query routing.
Routes user queries to: coder | image | rag (default).
"""

from __future__ import annotations

import logging
import re
from typing import Literal

logger = logging.getLogger(__name__)

Intent = Literal["coder", "image", "rag"]

_CODER_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\b(write|create|build|implement|develop|generate|make)\b.{0,40}\b(code|function|class|script|program|app|module|snippet|algorithm)\b",
        r"\b(debug|fix|refactor|optimize)\b.{0,40}\b(code|function|bug|error|exception)\b",
        r"\b(python|javascript|typescript|java|c\+\+|c#|rust|go|kotlin|swift|bash|sql)\b.{0,60}\b(code|function|class|script|program|example)\b",
        r"\b(sort|search|traverse|parse|tokenize|serialize|deserialize|encrypt|decrypt)\b.{0,40}\b(algorithm|function|code)\b",
        r"\bhow\b.{0,20}\b(implement|code|program|write)\b",
        r"\b(fibonacci|factorial|binary\s*search|bubble\s*sort|quicksort|merge\s*sort|linked\s*list|binary\s*tree|graph\s*traversal)\b",
        r"\bping\s*pong\b",
        r"\b(calculator|snake\s*game|tic\s*tac\s*toe|chess\s*board|todo\s*app)\b.{0,30}\b(code|python|script|program)\b",
        r"\b(function|class|method|variable|loop|recursion|api|endpoint|database\s*query)\b.{0,20}\b(in|using|with)\b.{0,20}\b(python|javascript|java|c\+\+|rust)\b",
        r"^(write|create|build|implement|code|program)\b.{0,80}\b(in python|in javascript|in java|using python)\b",
        r"\b(unit test|test case|mock|stub|fixture)\b.{0,30}\b(python|javascript|java|code)\b",
    ]
]

_IMAGE_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\b(draw|paint|sketch|illustrate|render)\b.{0,50}\b(image|picture|photo|art|illustration|portrait|landscape)\b",
        r"\b(generate|create|make|produce)\b.{0,50}\b(image|picture|photo|art|illustration|painting|drawing)\b",
        r"\bimage\s+of\b",
        r"\bpicture\s+of\b",
        r"\bphoto\s+of\b",
        r"\ba\s+(bird|cat|dog|horse|mountain|ocean|forest|sunset|city|car|flower)\b",
        r"\bvisual(i[sz]e|ization)?\b",
        r"\b(portrait|landscape|artwork|wallpaper|thumbnail)\b.{0,30}\b(of|with|showing)\b",
    ]
]

_CODER_STRONG = re.compile(
    r"^\s*(write|create|implement|build)\b.{0,100}$|"
    r"\bin python\b|\bin javascript\b|\bin java\b|\bin c\+\+\b",
    re.IGNORECASE,
)


def classify(query: str) -> Intent:
    """
    Classify a user query into one of: 'coder', 'image', 'rag'.
    Falls back to 'rag' when intent is ambiguous.
    """
    q = str(query or "").strip()
    if not q:
        return "rag"

    coder_hits = sum(1 for pat in _CODER_PATTERNS if pat.search(q))
    image_hits = sum(1 for pat in _IMAGE_PATTERNS if pat.search(q))
    strong_coder = bool(_CODER_STRONG.search(q))

    if strong_coder or coder_hits >= 2:
        logger.debug("[CLASSIFIER] → CODER (coder_hits=%d, strong=%s)", coder_hits, strong_coder)
        return "coder"
    if image_hits >= 1 and coder_hits == 0:
        logger.debug("[CLASSIFIER] → IMAGE (image_hits=%d)", image_hits)
        return "image"
    if coder_hits == 1:
        logger.debug("[CLASSIFIER] → CODER (single hit)")
        return "coder"

    logger.debug("[CLASSIFIER] → RAG (default, coder=%d image=%d)", coder_hits, image_hits)
    return "rag"


class IntentClassifier:
    """Stateless intent classifier. Thread-safe."""

    def classify(self, query: str) -> Intent:
        return classify(query)

"""Language mapping utilities for PaddleOCR plugin."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

_LANG_PATH = Path(__file__).with_name("languages.json")


def _load_language_map(path: Path = _LANG_PATH) -> Dict[str, str]:
    """Load language map from JSON; fallback to baked defaults if missing/invalid."""
    fallback = {
        "eng": "en",
        "chi_sim": "ch",
        "chi_tra": "chinese_cht",
        "fra": "fr",
        "deu": "german",
        "jpn": "japan",
        "kor": "korean",
        "spa": "spanish",
        "rus": "ru",
        "ara": "ar",
        "hin": "hi",
        "por": "pt",
        "ita": "it",
        "tur": "tr",
        "vie": "vi",
        "tha": "th",
    }
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in data.items()):
            return data
    except OSError:
        pass
    except json.JSONDecodeError:
        pass
    return fallback


LANGUAGE_MAP = _load_language_map()


def to_paddle_lang(options) -> str:
    """Convert OCRmyPDF language list to PaddleOCR language code."""
    langs = getattr(options, "languages", None) or []
    if not langs:
        return "en"
    lang = str(langs[0]).lower()
    return LANGUAGE_MAP.get(lang, lang)


def to_hocr_lang(paddle_lang: str) -> str:
    """Return hOCR lang code matching the original Tesseract-style code."""
    reverse = {v: k for k, v in LANGUAGE_MAP.items()}
    return reverse.get(paddle_lang, paddle_lang)

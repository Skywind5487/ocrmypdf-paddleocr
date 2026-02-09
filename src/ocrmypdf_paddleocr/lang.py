"""Language mapping utilities for PaddleOCR plugin."""

from __future__ import annotations

LANGUAGE_MAP = {
    'eng': 'en',
    'chi_sim': 'ch',
    'chi_tra': 'chinese_cht',
    'fra': 'fr',
    'deu': 'german',
    'jpn': 'japan',
    'kor': 'korean',
    'spa': 'spanish',
    'rus': 'ru',
    'ara': 'ar',
    'hin': 'hi',
    'por': 'pt',
    'ita': 'it',
    'tur': 'tr',
    'vie': 'vi',
    'tha': 'th',
}


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

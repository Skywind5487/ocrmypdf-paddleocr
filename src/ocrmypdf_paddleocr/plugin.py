"""PaddleOCR engine plugin for OCRmyPDF."""

from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image

from ocrmypdf import hookimpl
from ocrmypdf.pluginspec import OcrEngine, OrientationConfidence

from . import engine, hocr, lang, options

log = logging.getLogger(__name__)


@hookimpl
def add_options(parser):
    return options.add_options(parser)


@hookimpl
def check_options(opts):
    return options.check_options(opts)


class PaddleOCREngine(OcrEngine):
    """Implements OCR with PaddleOCR, delegating to submodules."""

    _ocr_cache = engine._ocr_cache
    _page_counter = 0
    LANGUAGE_MAP = lang.LANGUAGE_MAP

    @staticmethod
    def version():
        """Return PaddleOCR version."""
        try:
            import paddleocr

            return paddleocr.__version__
        except (ImportError, AttributeError):
            return "2.7.0"

    @staticmethod
    def creator_tag(options):
        return f"PaddleOCR {PaddleOCREngine.version()}"

    def __str__(self):
        return f"PaddleOCR {PaddleOCREngine.version()}"

    @staticmethod
    def languages(options):
        return {
            "en",
            "ch",
            "chinese_cht",
            "ta",
            "te",
            "ka",
            "latin",
            "ar",
            "cy",
            "da",
            "de",
            "es",
            "et",
            "fr",
            "ga",
            "hi",
            "it",
            "ja",
            "ko",
            "la",
            "nl",
            "no",
            "oc",
            "pt",
            "ro",
            "ru",
            "sr",
            "sv",
            "tr",
            "uk",
            "vi",
            # Tesseract codes for compatibility
            "eng",
            "chi_sim",
            "chi_tra",
            "deu",
            "fra",
            "spa",
            "rus",
            "jpn",
            "kor",
        }

    @staticmethod
    def _get_paddle_lang(options):
        return lang.to_paddle_lang(options)

    @staticmethod
    def _get_paddle_ocr(options):
        return engine.make_engine(options)

    @staticmethod
    def get_orientation(input_file: Path, options) -> OrientationConfidence:
        return OrientationConfidence(angle=0, confidence=0.0)

    @staticmethod
    def get_deskew(input_file: Path, options) -> float:
        return 0.0

    @staticmethod
    def generate_hocr(input_file: Path, output_hocr: Path, output_text: Path, options):
        return hocr.generate_hocr(input_file, output_hocr, output_text, options)

    @staticmethod
    def generate_pdf(input_file: Path, output_pdf: Path, output_text: Path, options):
        log.debug(f"Generating PDF from {input_file}")
        output_hocr = output_pdf.with_suffix(".hocr")
        PaddleOCREngine.generate_hocr(input_file, output_hocr, output_text, options)

        from ocrmypdf.hocrtransform import HocrTransform

        with Image.open(input_file) as img:
            dpi = img.info.get("dpi", (300, 300))[0]

        hocr_transform = HocrTransform(hocr_filename=output_hocr, dpi=dpi)
        hocr_transform.to_pdf(
            out_filename=output_pdf,
            image_filename=input_file,
            invisible_text=True,
        )


@hookimpl
def get_ocr_engine():
    return PaddleOCREngine()


"""hOCR generation helpers."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image

from . import engine
from .lang import to_hocr_lang, to_paddle_lang

log = logging.getLogger(__name__)


def generate_hocr(input_file: Path, output_hocr: Path, output_text: Path, options):
    """Generate hOCR output for an image using a PaddleOCR engine."""
    log.debug(f"Running PaddleOCR on {input_file}")

    paddle_ocr = engine.make_engine(options)

    with Image.open(input_file) as img:
        width, height = img.size
        dpi = img.info.get('dpi', (300, 300))
        log.debug(f"Input image: {width}x{height}, DPI: {dpi}")

    result = paddle_ocr.predict(
        str(input_file),
        return_word_box=getattr(options, 'paddle_return_word_box', True),
    )

    scale_x = 1.0
    scale_y = 1.0
    if result and len(result) > 0:
        ocr_result = result[0]
        if hasattr(ocr_result, 'get'):
            doc_prep_res = ocr_result.get('doc_preprocessor_res')
            if doc_prep_res and hasattr(doc_prep_res, 'get'):
                preprocessed_img = doc_prep_res.get('output_img')
                if preprocessed_img is not None:
                    if isinstance(preprocessed_img, np.ndarray):
                        prep_height, prep_width = preprocessed_img.shape[:2]
                        scale_x = width / prep_width
                        scale_y = height / prep_height
                        log.debug(
                            f"Preprocessed image: {prep_width}x{prep_height}, "
                            f"scaling factors: x={scale_x:.4f}, y={scale_y:.4f}"
                        )

    lang = to_paddle_lang(options)
    hocr_lang = to_hocr_lang(lang) or 'eng'

    hocr_lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN"',
        '    "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">',
        f'<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="{hocr_lang}" lang="{hocr_lang}">',
        '<head>',
        '<title></title>',
        '<meta http-equiv="content-type" content="text/html; charset=utf-8" />',
        '<meta name="ocr-system" content="PaddleOCR via ocrmypdf-paddleocr" />',
        '<meta name="ocr-capabilities" content="ocr_page ocr_carea ocr_par ocr_line ocrx_word" />',
        '</head>',
        '<body>',
        f'<div class="ocr_page" id="page_1" title="bbox 0 0 {width} {height}">',
    ]

    all_text = []

    if result and len(result) > 0:
        ocr_result = result[0]

        texts = ocr_result.get('rec_texts', [])
        scores = ocr_result.get('rec_scores', [])
        polys = ocr_result.get('rec_polys', [])
        text_words = ocr_result.get('text_word', [])
        text_word_regions = ocr_result.get('text_word_region', [])

        has_word_boxes = bool(text_words and text_word_regions)
        log.debug(f"PaddleOCR found {len(texts)} text regions, word boxes: {has_word_boxes}")

        word_id = 1
        carea_id = 1
        par_id = 1

        for line_id, (text, score, poly) in enumerate(zip(texts, scores, polys), 1):
            if not text:
                continue

            all_text.append(text)

            if isinstance(poly, np.ndarray):
                poly_scaled = poly * [scale_x, scale_y]
                x_min = int(poly_scaled[:, 0].min())
                x_max = int(poly_scaled[:, 0].max())
                if len(poly_scaled) == 4:
                    y_min = int((poly_scaled[0][1] + poly_scaled[1][1]) / 2)
                    y_max = int((poly_scaled[2][1] + poly_scaled[3][1]) / 2)
                else:
                    y_min = int(poly_scaled[:, 1].min())
                    y_max = int(poly_scaled[:, 1].max())
            else:
                xs = [int(point[0] * scale_x) for point in poly]
                ys = [int(point[1] * scale_y) for point in poly]
                x_min, y_min, x_max, y_max = min(xs), min(ys), max(xs), max(ys)

            conf_pct = int(score * 100)

            hocr_lines.append(
                f'<div class="ocr_carea" id="carea_{carea_id}" title="bbox {x_min} {y_min} {x_max} {y_max}">'
            )
            hocr_lines.append(
                f'<p class="ocr_par" id="par_{par_id}" lang="{hocr_lang}" title="bbox {x_min} {y_min} {x_max} {y_max}">'
            )
            hocr_lines.append(
                f'<span class="ocr_line" id="line_{line_id}" '
                f'title="bbox {x_min} {y_min} {x_max} {y_max}; baseline 0 0; x_wconf {conf_pct}">'
            )

            line_idx = line_id - 1
            if (
                has_word_boxes
                and line_idx < len(text_words)
                and line_idx < len(text_word_regions)
                and text_words[line_idx]
                and text_word_regions[line_idx]
            ):
                line_word_tokens = text_words[line_idx]
                line_word_boxes = text_word_regions[line_idx]

                merged_words = []
                current_word = []
                current_boxes = []

                for token, box in zip(line_word_tokens, line_word_boxes):
                    token_str = str(token).strip()
                    if not token_str or token_str.isspace():
                        if current_word:
                            merged_words.append((''.join(current_word), current_boxes))
                            current_word = []
                            current_boxes = []
                    else:
                        current_word.append(token_str)
                        current_boxes.append(box)

                if current_word:
                    merged_words.append((''.join(current_word), current_boxes))

                for i, (word, boxes) in enumerate(merged_words):
                    if not word:
                        continue

                    all_xs = []
                    all_ys_top = []
                    all_ys_bottom = []

                    for box in boxes:
                        if isinstance(box, np.ndarray):
                            box_scaled = box * [scale_x, scale_y]
                            all_xs.extend(box_scaled[:, 0])
                            if len(box_scaled) == 4:
                                all_ys_top.append((box_scaled[0][1] + box_scaled[1][1]) / 2)
                                all_ys_bottom.append((box_scaled[2][1] + box_scaled[3][1]) / 2)
                            else:
                                all_ys_top.append(box_scaled[:, 1].min())
                                all_ys_bottom.append(box_scaled[:, 1].max())
                        else:
                            for point in box:
                                all_xs.append(point[0] * scale_x)
                                all_ys_top.append(point[1] * scale_y)
                                all_ys_bottom.append(point[1] * scale_y)

                    word_x_min = int(min(all_xs))
                    word_x_max = int(max(all_xs))
                    word_y_min = int(min(all_ys_top))
                    word_y_max = int(max(all_ys_bottom))

                    word_escaped = (
                        word.replace('&', '&amp;')
                        .replace('<', '&lt;')
                        .replace('>', '&gt;')
                    )

                    hocr_lines.append(
                        f'<span class="ocrx_word" id="word_{word_id}" '
                        f'title="bbox {word_x_min} {word_y_min} {word_x_max} {word_y_max}; '
                        f'x_wconf {conf_pct}">{word_escaped}</span>'
                    )

                    if i < len(merged_words) - 1:
                        hocr_lines.append(' ')

                    word_id += 1
            else:
                words = text.split()
                if words:
                    line_width = x_max - x_min
                    total_chars = sum(len(w) for w in words)
                    num_spaces = len(words) - 1
                    if total_chars + num_spaces > 0:
                        total_space_width = line_width - total_chars * (line_width / (total_chars + num_spaces))
                        space_width = int(total_space_width / num_spaces) if num_spaces > 0 else 0
                    else:
                        space_width = 0
                    word_area_width = line_width - (space_width * num_spaces)

                    current_x = x_min
                    for i, word in enumerate(words):
                        if total_chars > 0:
                            word_width = int(word_area_width * len(word) / total_chars)
                        else:
                            word_width = line_width // len(words)

                        if i == len(words) - 1:
                            word_x_max = x_max
                        else:
                            word_x_max = current_x + word_width

                        word_escaped = (
                            word.replace('&', '&amp;')
                            .replace('<', '&lt;')
                            .replace('>', '&gt;')
                        )

                        hocr_lines.append(
                            f'<span class="ocrx_word" id="word_{word_id}" '
                            f'title="bbox {current_x} {y_min} {word_x_max} {y_max}; '
                            f'x_wconf {conf_pct}">{word_escaped}</span>'
                        )

                        if i < len(words) - 1:
                            hocr_lines.append(' ')

                        current_x = word_x_max + space_width
                        word_id += 1

            hocr_lines.append('</span>')
            hocr_lines.append('</p>')
            hocr_lines.append('</div>')

            carea_id += 1
            par_id += 1

    hocr_lines.extend([
        '</div>',
        '</body>',
        '</html>',
    ])

    output_hocr.write_text('\n'.join(hocr_lines), encoding='utf-8')
    text_content = '\n'.join(all_text)
    output_text.write_text(text_content, encoding='utf-8')
    log.debug(f"Generated hOCR with {len(all_text)} text regions")

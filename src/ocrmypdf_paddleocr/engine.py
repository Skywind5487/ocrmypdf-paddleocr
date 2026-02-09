"""Engine factory and cache for PaddleOCR."""

from __future__ import annotations

import logging
import os

from .lang import to_paddle_lang

try:
    from paddleocr import PaddleOCR
except ImportError:
    PaddleOCR = None

log = logging.getLogger(__name__)

_ocr_cache = {}


def make_engine(options):
    """Create (or reuse) a configured PaddleOCR instance."""
    # OCRmyPDF Tesseract plugin may set OMP_THREAD_LIMIT; Paddle needs more threads.
    saved_omp_limit = os.environ.get('OMP_THREAD_LIMIT')
    if saved_omp_limit:
        log.warning(f"Removing OMP_THREAD_LIMIT={saved_omp_limit} set by Tesseract plugin")
        os.environ.pop('OMP_THREAD_LIMIT', None)

    paddle_lang = to_paddle_lang(options)
    log.debug(f"Initializing PaddleOCR with language: {paddle_lang}")

    use_gpu = getattr(options, 'paddle_use_gpu', False)
    try:
        import paddle  # type: ignore

        if use_gpu and hasattr(paddle, "device") and not paddle.device.is_compiled_with_cuda():
            log.warning("Requested GPU but Paddle is not compiled with CUDA; falling back to CPU.")
            use_gpu = False
    except ImportError:
        pass

    kwargs = {
        'use_textline_orientation': getattr(options, 'paddle_use_angle_cls', True),
        'lang': paddle_lang,
        'use_doc_unwarping': False,
        'use_doc_orientation_classify': False,
    }

    kwargs['device'] = 'gpu' if use_gpu else 'cpu'

    if getattr(options, 'paddle_det_model_dir', None):
        kwargs['text_detection_model_dir'] = options.paddle_det_model_dir
    if getattr(options, 'paddle_rec_model_dir', None):
        kwargs['text_recognition_model_dir'] = options.paddle_rec_model_dir
    if getattr(options, 'paddle_cls_model_dir', None):
        kwargs['textline_orientation_model_dir'] = options.paddle_cls_model_dir
    if getattr(options, 'paddle_det_limit_side_len', None):
        kwargs['text_det_limit_side_len'] = options.paddle_det_limit_side_len
    if getattr(options, 'paddle_det_input_shape', None):
        try:
            kwargs['text_det_input_shape'] = [int(v.strip()) for v in str(options.paddle_det_input_shape).split(',')]
        except ValueError:
            log.warning(
                f"Ignoring invalid --paddle-det-input-shape={options.paddle_det_input_shape!r}"
            )

    cache_key = (
        kwargs.get('device'),
        kwargs.get('lang'),
        kwargs.get('text_detection_model_dir'),
        kwargs.get('text_recognition_model_dir'),
        kwargs.get('textline_orientation_model_dir'),
        kwargs.get('text_det_limit_side_len'),
        tuple(kwargs.get('text_det_input_shape', [])),
    )

    cached = _ocr_cache.get(cache_key)
    if cached is not None:
        return cached

    log.debug(f"Creating PaddleOCR with kwargs: {kwargs}")
    created = PaddleOCR(**kwargs)
    _ocr_cache[cache_key] = created
    return created

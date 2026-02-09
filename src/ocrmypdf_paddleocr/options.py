"""CLI option helpers for PaddleOCR plugin."""

from __future__ import annotations

from ocrmypdf import hookimpl

from ocrmypdf.exceptions import MissingDependencyError

try:
    from paddleocr import PaddleOCR  # noqa: F401
except (ImportError, AttributeError):
    PaddleOCR = None


@hookimpl
def add_options(parser):
    """Add PaddleOCR-specific options to the argument parser."""
    paddle = parser.add_argument_group(
        "PaddleOCR",
        "Options for PaddleOCR engine"
    )
    paddle.add_argument(
        '--paddle-use-gpu',
        action='store_true',
        help='Use GPU acceleration for PaddleOCR (requires GPU-enabled PaddlePaddle)',
    )
    paddle.add_argument(
        '--paddle-no-angle-cls',
        action='store_false',
        dest='paddle_use_angle_cls',
        default=True,
        help='Disable text orientation classification',
    )
    paddle.add_argument(
        '--paddle-show-log',
        action='store_true',
        help='Show PaddleOCR internal logging',
    )
    paddle.add_argument(
        '--paddle-det-limit-side-len',
        type=int,
        default=None,
        help='Limit text detection input side length to reduce memory usage (e.g. 1536)',
    )
    paddle.add_argument(
        '--paddle-det-input-shape',
        metavar='C,H,W',
        help='Override text detection input shape, e.g. 3,960,960',
    )
    paddle.add_argument(
        '--paddle-no-word-box',
        action='store_false',
        dest='paddle_return_word_box',
        default=True,
        help='Disable PaddleOCR word box output to reduce memory usage',
    )
    paddle.add_argument(
        '--paddle-det-model-dir',
        metavar='DIR',
        help='Path to text detection model directory',
    )
    paddle.add_argument(
        '--paddle-rec-model-dir',
        metavar='DIR',
        help='Path to text recognition model directory',
    )
    paddle.add_argument(
        '--paddle-cls-model-dir',
        metavar='DIR',
        help='Path to text orientation classification model directory',
    )


@hookimpl
def check_options(options):
    """Validate PaddleOCR options."""
    if PaddleOCR is None:
        raise MissingDependencyError(
            "PaddleOCR is not installed. "
            "Install it with: pip install paddlepaddle paddleocr"
        )

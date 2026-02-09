import importlib
import sys
import types
from argparse import Namespace

import pytest
from PIL import Image


class FakePaddleOCR:
    created = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.result = []
        self.predict_calls = []
        FakePaddleOCR.created.append(self)

    def predict(self, image_path, return_word_box=True):
        self.predict_calls.append((image_path, return_word_box))
        return self.result


@pytest.fixture
def fake_paddle_module(monkeypatch):
    module = types.SimpleNamespace(PaddleOCR=FakePaddleOCR)
    monkeypatch.setitem(sys.modules, "paddleocr", module)
    return FakePaddleOCR


@pytest.fixture
def plugin(fake_paddle_module, monkeypatch):
    import ocrmypdf_paddleocr.plugin as plugin_mod

    importlib.reload(plugin_mod)
    plugin_mod.PaddleOCREngine._ocr_cache.clear()
    plugin_mod.PaddleOCREngine._page_counter = 0
    return plugin_mod


@pytest.fixture
def blank_image(tmp_path):
    path = tmp_path / "page.png"
    Image.new("RGB", (100, 100), color="white").save(path)
    return path


@pytest.fixture
def dummy_options():
    def _opts(**kwargs):
        defaults = dict(
            paddle_use_gpu=False,
            paddle_use_angle_cls=True,
            paddle_det_model_dir=None,
            paddle_rec_model_dir=None,
            paddle_cls_model_dir=None,
            paddle_det_limit_side_len=None,
            paddle_det_input_shape=None,
            paddle_return_word_box=True,
            languages=["eng"],
        )
        defaults.update(kwargs)
        return Namespace(**defaults)

    return _opts

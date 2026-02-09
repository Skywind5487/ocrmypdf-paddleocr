def test_engine_respects_gpu_and_angle(plugin, dummy_options):
    opts = dummy_options(paddle_use_gpu=True, paddle_use_angle_cls=False)
    engine = plugin.PaddleOCREngine._get_paddle_ocr(opts)
    assert engine.kwargs["device"] == "gpu"
    assert engine.kwargs["use_textline_orientation"] is False
    if "use_doc_preprocessor" in engine.kwargs:
        assert engine.kwargs["use_doc_preprocessor"] is False


def test_engine_cache_key(plugin, dummy_options):
    opts1 = dummy_options(languages=["eng"])
    e1 = plugin.PaddleOCREngine._get_paddle_ocr(opts1)
    opts2 = dummy_options(languages=["eng"])
    e2 = plugin.PaddleOCREngine._get_paddle_ocr(opts2)
    assert e1 is e2  # same cache key

    opts3 = dummy_options(languages=["chi_tra"])
    e3 = plugin.PaddleOCREngine._get_paddle_ocr(opts3)
    assert e3 is not e1  # different lang -> new cache entry


def test_engine_invalid_det_input_shape_ignored(plugin, dummy_options):
    opts = dummy_options(paddle_det_input_shape="abc")
    engine = plugin.PaddleOCREngine._get_paddle_ocr(opts)
    assert "text_det_input_shape" not in engine.kwargs


def test_gpu_request_downgrades_without_cuda(monkeypatch, dummy_options):
    import importlib
    import types
    import sys
    import conftest

    class Dev:
        @staticmethod
        def is_compiled_with_cuda():
            return False

        @staticmethod
        def get_device():
            return "cpu"

    fake_paddle = types.SimpleNamespace(device=Dev())
    monkeypatch.setitem(sys.modules, "paddle", fake_paddle)
    monkeypatch.setitem(sys.modules, "paddleocr", types.SimpleNamespace(PaddleOCR=conftest.FakePaddleOCR))

    import ocrmypdf_paddleocr.engine as engine_mod

    importlib.reload(engine_mod)
    opts = dummy_options(paddle_use_gpu=True)
    eng = engine_mod.make_engine(opts)
    assert eng.kwargs["device"] == "cpu"

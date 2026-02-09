def test_engine_respects_gpu_and_angle(plugin, dummy_options):
    opts = dummy_options(paddle_use_gpu=True, paddle_use_angle_cls=False)
    engine = plugin.PaddleOCREngine._get_paddle_ocr(opts)
    assert engine.kwargs["device"] == "gpu"
    assert engine.kwargs["use_textline_orientation"] is False


def test_engine_cache_key(plugin, dummy_options):
    opts1 = dummy_options(languages=["eng"])
    e1 = plugin.PaddleOCREngine._get_paddle_ocr(opts1)
    opts2 = dummy_options(languages=["eng"])
    e2 = plugin.PaddleOCREngine._get_paddle_ocr(opts2)
    assert e1 is e2  # same cache key

    opts3 = dummy_options(languages=["chi_tra"])
    e3 = plugin.PaddleOCREngine._get_paddle_ocr(opts3)
    assert e3 is not e1  # different lang -> new cache entry

def test_language_map_forward_and_reverse(plugin):
    lang_map = plugin.PaddleOCREngine.LANGUAGE_MAP
    assert lang_map["chi_tra"] == "chinese_cht"
    assert plugin.PaddleOCREngine._get_paddle_lang(
        type("O", (), {"languages": ["chi_tra"]})
    ) == "chinese_cht"

    reverse = {v: k for k, v in lang_map.items()}
    assert reverse["chinese_cht"] == "chi_tra"
    # fallback: unknown stays the same
    assert plugin.PaddleOCREngine._get_paddle_lang(
        type("O", (), {"languages": ["xx"]})
    ) == "xx"

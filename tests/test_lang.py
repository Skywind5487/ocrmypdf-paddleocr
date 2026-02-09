import ocrmypdf_paddleocr.lang as lang


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


def test_language_map_loads_from_json(tmp_path, monkeypatch):
    json_path = tmp_path / "languages.json"
    json_path.write_text('{"foo":"bar"}', encoding="utf-8")
    loaded = lang._load_language_map(json_path)
    assert loaded["foo"] == "bar"


def test_language_map_invalid_json_falls_back(tmp_path, monkeypatch):
    bad_path = tmp_path / "bad.json"
    bad_path.write_text('{"ok": 1', encoding="utf-8")
    loaded = lang._load_language_map(bad_path)
    assert loaded["eng"] == "en"


def test_supported_languages_covers_map_keys_and_values():
    sup = lang.SUPPORTED_LANGUAGES
    for k, v in lang.LANGUAGE_MAP.items():
        assert k in sup
        assert v in sup

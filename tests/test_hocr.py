import numpy as np


def make_result(text="ABC"):
    poly = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
    return [
        {
            "rec_texts": [text],
            "rec_scores": [0.9],
            "rec_polys": [poly],
            "text_word": [[text[:2], text[2:]]],
            "text_word_region": [
                [
                    np.array([[0, 0], [5, 0], [5, 10], [0, 10]], dtype=float),
                    np.array([[5, 0], [10, 0], [10, 10], [5, 10]], dtype=float),
                ]
            ],
        }
    ]


def test_generate_hocr_with_word_boxes(plugin, dummy_options, blank_image, fake_paddle_module, tmp_path):
    opts = dummy_options(languages=["chi_tra"])
    engine = plugin.PaddleOCREngine._get_paddle_ocr(opts)
    engine.result = make_result("肌膚")

    hocr_path = tmp_path / "out.hocr"
    txt_path = tmp_path / "out.txt"
    plugin.PaddleOCREngine.generate_hocr(blank_image, hocr_path, txt_path, opts)

    hocr = hocr_path.read_text(encoding="utf-8")
    txt = txt_path.read_text(encoding="utf-8")

    assert 'xml:lang="chi_tra"' in hocr
    assert "bbox" in hocr and "x_wconf" in hocr
    assert "肌膚" in txt

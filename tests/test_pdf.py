import sys
import types
from pathlib import Path


def test_generate_pdf_calls_hocr_and_writes(plugin, dummy_options, blank_image, monkeypatch, tmp_path):
    opts = dummy_options()
    pdf = tmp_path / "out.pdf"
    txt = tmp_path / "out.txt"
    hocr_calls = []
    def fake_generate_hocr(input_file, output_hocr, output_text, options):
        hocr_calls.append((input_file, output_hocr, output_text))
        output_hocr.write_text("hocr", encoding="utf-8")
        output_text.write_text("txt", encoding="utf-8")
    monkeypatch.setattr(plugin.hocr, "generate_hocr", fake_generate_hocr)

    class FakeHocrTransform:
        def __init__(self, hocr_filename, dpi):
            self.hocr_filename = hocr_filename
            self.dpi = dpi
        def to_pdf(self, out_filename, image_filename, invisible_text=True):
            Path(out_filename).write_text("pdf", encoding="utf-8")
    monkeypatch.setitem(sys.modules, "ocrmypdf.hocrtransform", types.SimpleNamespace(HocrTransform=FakeHocrTransform))

    plugin.PaddleOCREngine.generate_pdf(blank_image, pdf, txt, opts)

    assert pdf.read_text(encoding="utf-8") == "pdf"
    assert hocr_calls and hocr_calls[0][0] == blank_image

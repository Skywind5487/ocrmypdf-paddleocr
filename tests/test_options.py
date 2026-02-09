import argparse


def test_add_options_defaults(plugin):
    parser = argparse.ArgumentParser()
    plugin.add_options(parser)
    opts = parser.parse_args([])
    assert opts.paddle_use_gpu is False
    assert opts.paddle_use_angle_cls is True
    assert opts.paddle_det_model_dir is None
    assert opts.paddle_rec_model_dir is None


def test_add_options_overrides(plugin):
    parser = argparse.ArgumentParser()
    plugin.add_options(parser)
    opts = parser.parse_args(
        [
            "--paddle-use-gpu",
            "--paddle-no-angle-cls",
            "--paddle-det-model-dir",
            "det",
            "--paddle-rec-model-dir",
            "rec",
            "--paddle-det-limit-side-len",
            "1024",
            "--paddle-det-input-shape",
            "3,640,640",
        ]
    )
    assert opts.paddle_use_gpu is True
    assert opts.paddle_use_angle_cls is False
    assert opts.paddle_det_model_dir == "det"
    assert opts.paddle_rec_model_dir == "rec"
    assert opts.paddle_det_limit_side_len == 1024
    assert opts.paddle_det_input_shape == "3,640,640"

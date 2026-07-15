from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

from rsr.common import parse_bool, parse_freegrasp_response, point_to_dataset_id
from rsr.run import _encode_image_like_smartgrasp, _save_numbered_png, _visible_instance_point


class CommonTests(unittest.TestCase):
    def test_parse_bool(self) -> None:
        self.assertTrue(parse_bool("TRUE"))
        self.assertFalse(parse_bool("false"))

    def test_parse_freegrasp_response(self) -> None:
        result = parse_freegrasp_response("[3, yellow propeller]", "fallback")
        self.assertEqual(result["selected_object_id"], 3)
        self.assertEqual(result["class_name"], "yellow propeller")
        result = parse_freegrasp_response("[pick object, 4, blue bolt]", "fallback")
        self.assertEqual(result["selected_object_id"], 4)

    def test_point_to_dataset_id(self) -> None:
        instances = np.array([[0, 1], [3, 0]], dtype=np.int32)
        self.assertEqual(
            point_to_dataset_id(instances, 0, 1),
            {"npz_label": 3, "dataset_object_id": 2},
        )
        self.assertEqual(
            point_to_dataset_id(instances, 0, 0),
            {"npz_label": 0, "dataset_object_id": None},
        )

    def test_visible_instance_point_is_inside_mask(self) -> None:
        instances = np.zeros((5, 5), dtype=np.int32)
        instances[0, 0:4] = 7
        instances[1:4, 0] = 7
        x, y = _visible_instance_point(instances, 7)
        self.assertEqual(int(instances[y, x]), 7)

    def test_numbered_png_preserves_resolution_and_png_format(self) -> None:
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "labeled.png"
            image = Image.new("RGB", (64, 48), "white")
            _save_numbered_png(
                image,
                [{"localization_id": 1, "x": 20, "y": 20}],
                path,
            )
            with Image.open(path) as labeled:
                self.assertEqual(labeled.format, "PNG")
                self.assertEqual(labeled.size, (64, 48))
                self.assertEqual(labeled.mode, "RGB")

    def test_smartgrasp_image_encoding_is_in_memory_jpeg(self) -> None:
        with TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.png"
            Image.new("RGB", (64, 48), "teal").save(source, format="PNG")
            encoded, details = _encode_image_like_smartgrasp(source)
            self.assertTrue(encoded.startswith("/9j/"))
            self.assertEqual(details["transport_format"], "JPEG")
            self.assertEqual(details["jpeg_quality"], 90)
            self.assertEqual(details["size"], [64, 48])
            self.assertFalse(details["persisted_to_disk"])


if __name__ == "__main__":
    unittest.main()

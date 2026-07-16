from __future__ import annotations

import hashlib
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

from rsr.common import parse_bool, parse_freegrasp_response, point_to_dataset_id
from rsr.run import (
    FREEGRASP_SYSTEM_PROMPT,
    _encode_original_png,
    _freegrasp_chat_payload,
    _save_freegrasp_matplotlib_png,
    _save_numbered_png,
    _visible_instance_point,
)


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

    def test_original_png_bytes_are_uploaded_without_conversion(self) -> None:
        with TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.png"
            Image.new("RGB", (64, 48), "teal").save(source, format="PNG")
            encoded, details = _encode_original_png(source)
            self.assertTrue(encoded.startswith("iVBOR"))
            self.assertEqual(details["transport_format"], "PNG")
            self.assertEqual(details["size"], [64, 48])
            self.assertTrue(details["transport_uses_source_bytes"])
            self.assertFalse(details["resized"])
            self.assertFalse(details["recompressed"])

    def test_chat_payload_matches_original_freegrasp_settings(self) -> None:
        self.assertEqual(len(FREEGRASP_SYSTEM_PROMPT), 1105)
        self.assertEqual(
            hashlib.sha256(FREEGRASP_SYSTEM_PROMPT.encode()).hexdigest(),
            "04b8cdfaee711bafaa126526ef46f470190baf2207e6a736e8419021889f0f77",
        )
        payload = _freegrasp_chat_payload("gpt-4o", "the plyer", "PNGDATA")
        self.assertEqual(payload["model"], "gpt-4o")
        self.assertEqual(payload["temperature"], 0)
        self.assertEqual(payload["max_tokens"], 713)
        self.assertEqual(payload["top_p"], 1)
        self.assertEqual(payload["frequency_penalty"], 0)
        self.assertEqual(payload["presence_penalty"], 0)
        self.assertEqual(payload["seed"], 0)
        self.assertEqual(
            payload["messages"][1]["content"][0],
            {"type": "text", "text": "Grasp the plyer"},
        )
        self.assertEqual(
            payload["messages"][1]["content"][1],
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,PNGDATA"},
            },
        )

    def test_freegrasp_matplotlib_prompt_is_high_resolution_rgba_png(self) -> None:
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "freegrasp.png"
            _save_freegrasp_matplotlib_png(
                Image.new("RGB", (120, 120), "white"),
                [{"localization_id": 1, "x": 60, "y": 60}],
                path,
            )
            with Image.open(path) as labeled:
                self.assertEqual(labeled.format, "PNG")
                self.assertEqual(labeled.mode, "RGBA")
                self.assertGreater(labeled.size[0], 120)
                self.assertGreater(labeled.size[1], 120)


if __name__ == "__main__":
    unittest.main()

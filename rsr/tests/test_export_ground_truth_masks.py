from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

from rsr.export_ground_truth_masks import export_instance_masks


class ExportGroundTruthMasksTests(unittest.TestCase):
    def test_exports_all_npz_labels_with_standard_gt_names(self) -> None:
        instances = np.array([[0, 1], [2, 2]], dtype=np.int32)
        with TemporaryDirectory() as temporary:
            records = export_instance_masks(instances, Path(temporary))
            self.assertEqual(
                [Path(item["path"]).name for item in records],
                ["mask_001_gt.png", "mask_002_gt.png"],
            )
            self.assertEqual(
                [item["dataset_object_id"] for item in records],
                [0, 1],
            )
            first = np.asarray(Image.open(records[0]["path"]).convert("L")) > 0
            np.testing.assert_array_equal(first, instances == 1)


if __name__ == "__main__":
    unittest.main()

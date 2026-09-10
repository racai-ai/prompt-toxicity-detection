import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models.pretrained import resolve_adapter_base_model_name_or_path


class TestPretrainedHelpers(unittest.TestCase):
    def test_resolve_adapter_base_model_name_or_path_uses_adapter_config_base_model(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(Path(tmpdir) / "adapter_config.json", "w", encoding="utf-8") as fh:
                json.dump({"base_model_name_or_path": "repo/base-model"}, fh)

            self.assertEqual(
                resolve_adapter_base_model_name_or_path(f"disk://{tmpdir}"),
                "repo/base-model",
            )

    def test_resolve_adapter_base_model_name_or_path_falls_back_to_local_path_without_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self.assertEqual(
                resolve_adapter_base_model_name_or_path(f"disk://{tmpdir}"),
                tmpdir,
            )


if __name__ == "__main__":
    unittest.main()

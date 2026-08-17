import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.download_pretrained_models import artifact_files
from scripts.verify_artifacts import FOLDS, MODEL_DIRECTORIES, REACTIONS, validate

class ArtifactToolTests(unittest.TestCase):
    def test_split_artifact_parts_keep_manifest_order(self):
        item = {
            "name": "split",
            "filename": "models.tar.gz",
            "parts": [
                {"filename": "models.part-00", "size": 1, "sha256": "a", "url": "u0"},
                {"filename": "models.part-01", "size": 1, "sha256": "b", "url": "u1"},
            ],
        }
        self.assertEqual([x["filename"] for x in artifact_files(item)],
                         ["models.part-00", "models.part-01"])

    def test_inference_validation_does_not_require_oof(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            avalon = root / "model_training/ckpt-searchfp/AvalonFingerprint_lgbm"
            optuna = root / "model_training/ckpt-optuna"
            for rxn in REACTIONS:
                for fold in FOLDS:
                    path = avalon / f"rxn_{rxn}/lgbm_fold{fold}.txt"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text("model")
                    for name in MODEL_DIRECTORIES:
                        path = optuna / name / f"rxn_{rxn}/lgbm_fold{fold}.txt"
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_text("model")
            with patch("scripts.verify_artifacts.ROOT", root):
                self.assertEqual(validate(mode="inference"), [])
                self.assertTrue(validate(mode="full"))

if __name__ == "__main__":
    unittest.main()

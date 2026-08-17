import importlib
import tempfile
import unittest
from pathlib import Path

from feature_generation.common import (
    calculate_manuscript_rdkit_descriptors,
    load_rdkit_descriptor_names,
    protected_outputs,
)

MODULES = [
    "mordred_gen", "mordred_gen_dict", "morfeus_atom_all", "morfeus_atom_rx1",
    "morfeus_qmdesc", "morfeus_qmdesc_round2_test", "psikit_qmdesc",
    "psikit_qmdesc_additives", "psikit_qmdesc_solv",
]

class FeatureGenerationSafetyTests(unittest.TestCase):
    def test_manuscript_rdkit_schema_is_fixed_at_210(self):
        from rdkit import Chem
        names = load_rdkit_descriptor_names()
        values = calculate_manuscript_rdkit_descriptors(Chem.MolFromSmiles("C"), names)
        self.assertEqual(len(names), 210)
        self.assertEqual(len(values), 210)

    def test_modules_import_without_running_generation(self):
        for name in MODULES:
            importlib.import_module(f"feature_generation.{name}")

    def test_existing_output_is_rejected_and_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "feature.csv"
            output.write_text("original")
            with self.assertRaises(FileExistsError):
                protected_outputs([output], overwrite=False)
            self.assertEqual(output.read_text(), "original")
            protected_outputs([output], overwrite=True)
            self.assertEqual(output.read_text(), "original")

if __name__ == "__main__":
    unittest.main()

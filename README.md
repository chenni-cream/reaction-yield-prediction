# Reaction Yield Prediction via Multi-Scale Feature Fusion

Source code and reproducibility artifacts for predicting reaction yields across eight reaction classes with molecular fingerprints, RDKit descriptors, solvent properties, quantum-chemical descriptors, and LightGBM.

## Manuscript model

The reported final model is the predefined arithmetic mean of three component models:

1. Avalon fingerprint;
2. Avalon fingerprint + selected RDKit descriptors;
3. Layered fingerprint + selected RDKit, solvent, and QC descriptors.

Adaptive reaction-specific weights are evaluated only as an exploratory training-set OOF analysis. External-test labels are not used to fit weights or select the final ensemble.

## Reproducibility options

### Level 1: recalculate the reported metrics

No pretrained models are required:

```bash
python scripts/recalculate_final_metrics.py
```

Expected summary:

```text
Macro Test R²: 0.417986
Macro Test MAE: 0.125919
Pooled Test R²: 0.439021
```

### Level 2: download pretrained models and rerun inference

The published `v1.0.0-manuscript` Release contains the three inference model groups and
the precomputed Product RDKit descriptors required to reproduce the released predictions:

```bash
python scripts/download_pretrained_models.py --inference-only
python model_training/6_test_evaluation/evaluate-final-simple-ensemble.py
```

To include the exploratory OOF comparison:

```bash
python model_training/5_ensemble_shap/ensemble-oof-analysis.py
```

Validate existing local artifacts with:

```bash
python scripts/verify_artifacts.py --mode inference --load-models
# Include OOF predictions and feature-importance files:
python scripts/verify_artifacts.py --mode full --load-models
```

### Level 3: train the final models from scratch

The repository includes the precomputed feature tables used by the final training pipeline. Run:

```bash
# Stage 2: train the final Avalon component models
python model_training/2_fingerprint_comparison/fingerprint-comparison.py

# Stage 4: train the two optimized fusion models and generate all OOF predictions
python model_training/4_optimization/optuna-tune-extended.py

# Stage 5: exploratory OOF comparison
python model_training/5_ensemble_shap/ensemble-oof-analysis.py

# Stage 6: final predefined simple-average external evaluation
python model_training/6_test_evaluation/evaluate-final-simple-ensemble.py
```

Stages 1 and 3 contain model-family, fingerprint, feature-fusion, and ablation experiments supporting the manuscript, but are not required to rebuild the three final component models.

## Optional feature regeneration

The final training workflow uses the committed precomputed feature tables. To regenerate features from molecular inputs, all feature scripts now use repository-relative defaults, expose command-line parameters, and refuse to overwrite existing outputs unless `--overwrite` is explicitly supplied.

Examples:

```bash
# Per-reaction RDKit descriptor matrices
python feature_generation/mordred_gen.py --data-dir data --output-dir /path/to/new/rdkit-output

# Per-component RDKit lookup tables
python feature_generation/mordred_gen_dict.py --column Reactant1 --output-dir /path/to/new/lookups

# Morfeus molecular descriptors
python feature_generation/morfeus_qmdesc.py --column Additive --output-dir /path/to/new/morfeus-output

# Round 2 test descriptors; names and indices 0-4 are accepted
python feature_generation/morfeus_qmdesc_round2_test.py --column Reactant1 --output-dir /path/to/new/test-output

# Atom-level descriptors
python feature_generation/morfeus_atom_all.py --column Reactant2 --output-dir /path/to/new/atom-output
python feature_generation/morfeus_atom_rx1.py --reaction 1 --data-dir data

# Psi4/Psikit descriptors
python feature_generation/psikit_qmdesc.py --column Reactant2 --output-dir /path/to/new/psikit-output
python feature_generation/psikit_qmdesc_additives.py --output-dir /path/to/new/psikit-output
python feature_generation/psikit_qmdesc_solv.py --output-dir /path/to/new/psikit-output
```

Use a new output directory for validation runs. `--overwrite` is intentionally required to replace an existing feature file. Morfeus/xTB and Psi4/Psikit calculations require their respective external runtimes and can be computationally expensive.

## Expected checkpoint layout

```text
model_training/
├── ckpt-searchfp/AvalonFingerprint_lgbm/
└── ckpt-optuna/
    ├── Avalon_FP/
    ├── Avalon_RDKit/
    └── Layered_RDKit_QC_Solvent/
```

Checkpoint directories and the large Product descriptor lookup are intentionally excluded
from normal Git history. They are downloaded from the GitHub Release by
`scripts/download_pretrained_models.py`; see `CODE_RELEASE_PLAN.md` for the packaging procedure.

## Results

The authoritative released result files are:

```text
model_training/results/final_simple_ensemble_metrics.csv
model_training/results/final_simple_ensemble_predictions.csv
model_training/results/component_model_external_metrics.csv
model_training/results/ensemble_oof_analysis.json
model_training/results/oof_ablation.csv
model_training/results/oof_ablation_grouped_bar.png
model_training/results/test_set_comparison_bar.png
```

`oof_ablation_grouped_bar.png` visualizes the exploratory training-set OOF comparison. `test_set_comparison_bar.png` visualizes external-test component-model and predefined simple-average performance. Both figures are generated by `notebooks/ablation_plot.ipynb`.

## Data and features

The `data/` directory contains the train/test tables and precomputed descriptor data used by the manuscript pipeline. See `data/README.md` for the inventory, source provenance, and third-party attribution notes.

Feature-generation scripts are retained to document descriptor construction. Several quantum-chemical scripts require external runtimes such as xTB or Psi4 and are substantially more expensive than model training. Until their command-line refactor and clean-environment validation are complete, the committed precomputed features are the supported reproduction input.

## Environment

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

The exact core Python environment used for the release checks is recorded in `requirements.txt`; `environment.yml` provides the equivalent Conda entry point. Optional QC feature generation and historical model-comparison or SHAP experiments use `requirements-optional.txt`; xTB and Psi4 must still be installed separately. The optional `scikit-fingerprints` package is not installed because its current RDKit constraint conflicts with the validated core environment; the fingerprint-comparison script uses its documented SECFP fallback in this environment. The fixed 210-name RDKit schema prevents descriptors added by newer RDKit releases from changing the manuscript feature layout.

## Lightweight validation

```bash
python -m compileall -q feature_generation model_training scripts tests
python scripts/recalculate_final_metrics.py
python -m unittest discover -s tests -v
```

The GitHub Actions workflow runs only lightweight checks and does not download multi-gigabyte model artifacts.

## License

The source code is released under the MIT License; see `LICENSE`. Dataset provenance and third-party attribution notes are documented in `data/README.md`.

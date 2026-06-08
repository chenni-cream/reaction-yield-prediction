# Reaction Yield Prediction via Multi-Scale Feature Fusion

Source code for predicting chemical reaction yields using a multi-scale feature fusion framework with LightGBM. This framework integrates molecular fingerprints, RDKit descriptors, quantum mechanical (QM) descriptors, and solvent properties to predict yields across 8 types of catalytic reactions.

## Features

Four categories of molecular features are fused at the reaction level:

| Feature Type | Description |
|---|---|
| Molecular Fingerprint | Avalon / ECFP4 / Layered Fingerprint |
| RDKit Descriptors | Physicochemical properties (MW, LogP, TPSA, etc.) |
| QM Descriptors | Dispersion, SASA, XTB electronic parameters |
| Solvent Properties | Physical constants, MNSol parameters, DrugBank attributes |

Each reaction is represented by concatenating features of all 5 molecular components (Reactant1, Reactant2, Product, Additive, Solvent).

## Requirements

- numpy
- pandas
- scikit-learn
- lightgbm
- xgboost
- catboost
- rdkit
- morfeus-ml
- psikit
- optuna
- matplotlib
- tqdm
- shap
- networkx
- skfp

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Feature Generation

```bash
# Generate RDKit descriptors
python feature_generation/mordred_gen.py

# Generate QM descriptors
python feature_generation/morfeus_qmdesc.py

# Generate QM descriptors for test set (Round 2)
python feature_generation/morfeus_qmdesc_round2_test.py --col 0
```

### Model Training

```bash
# Stage 1: Model selection
python model_training/1_model_selection/layered_fp_model_comparison.py
python model_training/1_model_selection/avalon_fp_model_comparison.py
# Stage 2: Full feature fusion
python model_training/3_feature_fusion_ablation/Avalon-RDKit.py
python model_training/3_feature_fusion_ablation/Layer-RDKit.py 
python model_training/3_feature_fusion_ablation/layer-RDkit-solvent-qc.py
# Stage 3: Feature ablation
python model_training/3_feature_fusion_ablation/avalon-rdkit-ablation.py
python model_training/3_feature_fusion_ablation/layer-RDkit-solvent-qc-ablation.py
# Stage 4: Optuna hyperparameter tuning
python model_training/4_optimization/optuna-tune-extended.py
python model_training/4_optimization/optuna-tune-config-c.py
# Stage 5: Ensemble model
python model_training/5_ensemble_shap/ensemble-weighted.py

```

### Test Evaluation

```bash
# Ensemble model test evaluation
python model_training/5_ensemble_shap/ensemble-weighted.py
```

## Data

The `data/` directory contains:

- `round1_train_data.csv` / `round2_train_data.csv` — Training data with SMILES and yield values
- `round1_test_data.csv` / `round1_test_data_with_ans.csv` / `round2_test_data_with_ans.csv` — Test data
- `extra-rdkit/` — Pre-computed RDKit descriptors (per reaction type)
- `qm_desc-morfeus/` — Pre-computed QM descriptors
- `reactionmatch/` — Reaction SMARTS pattern matching results
- `solvents/` — Solvent property database
- `drugbank/` — DrugBank solvent attributes

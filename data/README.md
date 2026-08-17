# Data and derived features

This directory currently contains the data tables and precomputed features used by the manuscript experiments. The previous statement that the original datasets were not included was incorrect.

## Core reaction tables

```text
round1_train_data.csv
round2_train_data.csv
round1_test_data.csv
round1_test_data_with_ans.csv
round2_test_data_with_ans.csv
```

Expected molecular columns are `Reactant1`, `Reactant2`, `Product`, `Additive`, and `Solvent`; training and answer-containing evaluation tables also contain `Yield`. Round 2 tables contain reaction-type labels for classes 2–8, while Round 1 is treated as reaction type 1.

The manuscript currently cites the source dataset at:

https://figshare.com/articles/dataset/Demo_organic_reaction_dataset_used_in_Physical_Science_Track_of_the_2nd_World_AI4S_Prize_/30265270/1

The source link is provided for provenance. Users should consult the source record and the relevant third-party terms when reusing the core tables, including answer-containing test tables.

## Derived feature directories

- `extra-rdkit/`: precomputed RDKit descriptor matrices and SMILES lookups.
- `qm_desc-morfeus/`: precomputed QC descriptors used for training.
- `qm_desc-morfeus-round1-test/`: QC descriptors for the Round 1 test set.
- `qm_desc-morfeus-round2-test/`: QC descriptors for the Round 2 test set.
- `reactionmatch/`: reaction matching and atom-mapping-derived files.
- `solvents/`: solvent property tables, including MNSol-derived content.
- `drugbank/`: DrugBank-derived solvent attributes.

## Source and third-party attribution notes

The following sources and software should be considered when citing or reusing these data and derived features:

1. Redistribution terms for the competition/Figshare reaction data.
2. Redistribution and attribution requirements for MNSol-derived tables.
3. Redistribution and attribution requirements for DrugBank-derived data.
4. Whether derived descriptors may be redistributed under the source-data terms.
5. Required citations for RDKit, Morfeus, xTB, Psi4/Psikit, MNSol, and DrugBank.

Consult the original source terms for redistribution and reuse requirements.

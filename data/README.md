# Data

Due to file size limitations, the original datasets are not included in this repository.

## Expected Data Structure

Place your data files in this directory following the structure below:

```
data/
├── round1_train_data.csv
├── round2_train_data.csv
├── round1_test_data_with_ans.csv
├── round2_test_data_with_ans.csv
└── full_database.xml        # molecular database (optional)
```

Each CSV file should contain the following columns:
- `Reactant1` - SMILES string of reactant 1
- `Reactant2` - SMILES string of reactant 2
- `Product`   - SMILES string of product
- `Additive`  - SMILES string of additive
- `Solvent`   - SMILES string of solvent
- `Yield`     - reaction yield (target variable)

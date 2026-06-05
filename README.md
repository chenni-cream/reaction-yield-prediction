 # Reaction Yield Prediction via Multi-Scale Feature Fusion                                                    
                                                                                                                
  Source code for predicting chemical reaction yields using a multi-scale feature fusion framework with         
  LightGBM. This framework integrates molecular fingerprints, RDKit descriptors, quantum mechanical (QM)        
  descriptors, and solvent properties to predict yields across 8 types of catalytic reactions.                  
                                                                                                                
  ## Project Structure                                      
                                                                                                                
  ├── data/                          # Datasets and pre-computed features
  ├── feature_generation/            # Feature extraction scripts                                               
  │   ├── psikit_qmdesc.py           #   QM descriptors (SCF energy, HOMO/LUMO, dipole)                         
  │   ├── mordred_gen.py             #   RDKit molecular descriptors                                            
  │   ├── morfeus_qmdesc.py          #   Morfeus QM descriptors (dispersion, SASA, XTB)                         
  │   ├── morfeus_atom_rx1.py        #   Atom-level TSEI descriptors                                            
  │   └── utils/                     #   TSEI, mol_utils, Sterimol                                              
  │                                                                                                             
  ├── model_training/                # Model training & evaluation (by experimental stage)                      
  │   ├── 1_model_selection/         #   Stage 1: Model comparison (RF/XGB/CatBoost/LGBM)                       
  │   ├── 2_fingerprint_comparison/  #   Stage 2: Fingerprint type comparison                                   
  │   ├── 3_feature_fusion_ablation/ #   Stage 3: Feature fusion & ablation study                               
  │   ├── 4_optimization/            #   Stage 4: Optuna tuning & feature selection                             
  │   ├── 5_ensemble_shap/           #   Stage 5: Ensemble model & SHAP analysis                                
  │   └── 6_test_evaluation/         #   Stage 6: Test set evaluation                                           
  │                                                                                                             
  └── notebooks/                     # EDA, preprocessing, and visualization notebooks                          
                                                                                                                
  ## Features                                                                                                   
                                                                                                                
  Four categories of molecular features are fused at the reaction level:                                        
                                                                                                                
  | Feature Type | Description | Dimensions per molecule |                                                      
  |---|---|---|                                             
  | Molecular Fingerprint | Avalon / ECFP4 / Layered Fingerprint | 2048-bit |                                   
  | RDKit Descriptors | Physicochemical properties (MW, LogP, TPSA, etc.) | 210 |                               
  | QM Descriptors | Dispersion, SASA, XTB electronic parameters | 10 |                                         
  | Solvent Properties | Physical constants, MNSol parameters, DrugBank attributes | 31 |                       
                                                                                                                
  Each reaction is represented by concatenating features of all 5 molecular components (Reactant1, Reactant2,   
  Product, Additive, Solvent).                                                                                  
                                                                                                                
  ## Workflow                                                                                                   
   
  Stage 1 → Stage 2 → Stage 3 → Stage 4 → Stage 5 → Stage 6                                                     
  Model       Finger-     Feature     Hyper-      Ensemble    Test
  Selection   print       Fusion &    parameter   & SHAP      Evaluation                                        
              Comparison  Ablation    Tuning                                                                    
                                                                                                                
  1. **Model Selection**: Compare RF, XGBoost, CatBoost, and LightGBM with molecular fingerprints → LightGBM    
  selected                                                                                                      
  2. **Fingerprint Comparison**: Evaluate Avalon, ECFP4, Layered, MACCS, and other fingerprints → Layered
  Fingerprint selected                                                                                          
  3. **Feature Fusion & Ablation**: Incrementally add RDKit → Solvent → QM descriptors with Top-K feature
  ablation                                                                                                      
  4. **Optimization**: Optuna-based hyperparameter tuning and feature importance-based selection
  5. **Ensemble & SHAP**: Three-model weighted ensemble with SHAP-based interpretability analysis               
  6. **Test Evaluation**: Generalization validation on held-out test sets                                       
                                                                                                                
  ## Requirements                                                                                               
                                                                                                                
  numpy                                                                                                         
  pandas                                                    
  scikit-learn                                                                                                  
  lightgbm                                                                                                      
  xgboost
  catboost                                                                                                      
  rdkit                                                     
  morfeus-ml                                                                                                    
  psikit                                                    
  optuna                                                                                                        
  shap
  matplotlib                                                                                                    
  tqdm                                                      
  networkx
  skfp

  Install dependencies:

  ```bash
  pip install -r requirements.txt
                                                                                                                
  Usage
                                                                                                                
  Feature Generation                                        

  # Generate RDKit descriptors
  python feature_generation/mordred_gen.py
                                                                                                                
  # Generate QM descriptors                                                                                     
  python feature_generation/morfeus_qmdesc.py                                                                   
                                                                                                                
  # Generate QM descriptors for test set (Round 2)                                                              
  python feature_generation/morfeus_qmdesc_round2_test.py --col 0                                               
                                                                                                                
  Model Training                                                                                                
                                                                                                                
  # Stage 1: Model selection                                                                                    
  python model_training/1_model_selection/layered_fp_model_comparison.py                                        
                                                                                                                
  # Stage 3: Full feature fusion (Layered FP + RDKit + Solvent + QC)                                            
  python model_training/3_feature_fusion_ablation/layer-RDkit-solvent-qc.py                                     
                                                                                                                
  # Stage 4: Optuna hyperparameter tuning                                                                       
  python model_training/4_optimization/optuna-tune-extended.py                                                  
                                                                                                                
  Test Evaluation                                                                                               
                                                                                                                
  # Full model test evaluation                                                                                  
  python model_training/6_test_evaluation/layer-RDKit-qc-solvent-test.py                                        
                                                                                                                
  Data                                                                                                          
                                                                                                                
  The data/ directory contains:                                                                                 
                                                            
  - round1_train_data.csv / round2_train_data.csv — Training data with SMILES and yield values                  
  - round1_test_data.csv / round1_test_data_with_ans.csv / round2_test_data_with_ans.csv — Test data
  - extra-rdkit/ — Pre-computed RDKit descriptors (per reaction type)                                           
  - qm_desc-morfeus/ — Pre-computed QM descriptors                                                              
  - reactionmatch/ — Reaction SMARTS pattern matching results                                                   
  - solvents/ — Solvent property database                                                                       
  - drugbank/ — DrugBank solvent attributes                                                                     
                                                                                                                
                                                     
                                            

# configs for training the different DTI models

We have 4 different DTI tree models:
- Baseline DTI model
    - processes only a single drug and target feature (e.g. fingerprint features)
    - uses dot-product between both embeddings to predict a single DTI score (Y_pKd or Y_KIBA)
    - only fine-tunes on a single DTI benchmark dataset (DAVIS or KIBA) in a single split setting (random or cold-drug split)
- Multi-modal DTI model
    - takes multiple drug and target feature inputs that are aggregated into a single drug and target embedding (e.g. pre-computed foundation model embeddings)
    - likewise uses dot-product between both embeddings to predict a single DTI score
    - likewise only fine-tunes on a single DTI benchmark dataset
- Multi-output DTI model
    - takes only a single drug and target feature (like baseline)
    - uses a multi-output head to predict multiple DTI scores
    - trains on the combined dataset of multiple DTI benchmark datasets (Davis, KIBA, Metz and BindingDB_Kd/Ki)
    - fine-tunes on a single DTI benchmark dataset (like baseline; transferring weights)
- Full model
    - takes multiple drug and target features that are aggregated into a single drug and target embedding (like multi-input)
    - uses a multi-output head to predict multiple DTI scores (like multi-output)
    - uses contrastive learning & reconstruction objectives to pre-train each branch on unlabeled drug and target data
    - trains on the combined dataset of multiple DTI benchmark datasets (like multi-output)
    - fine-tunes on a single DTI benchmark dataset (like baseline; transferring weights)
  
Depending on which training phases are used, we have 3 different config types:
- `<DAVIS/KIBA>_<rand/cold>.yaml`: fine-tune the model on a single DTI benchmark dataset (DAVIS or KIBA) in a single split setting (random or cold-drug split)
- `pretrain_<rand/cold>.yaml`: train the model on the combined dataset of multiple DTI benchmark datasets (Davis, KIBA, Metz and BindingDB_Kd/Ki - w/out data leakage!)
- `pretrain_<drug/target>.yaml`: pre-train the drug or target branch on the unlabeled data

Additional notes
- There is both a top-level, and model-specific `common.yaml` file that contains configuration parameters that are shared with all downstream configs.
- Config files have a `gridsearch` field that contains parameters for gridsearch, these override the default values when running a gridsearch experiment with `--gridsearch`
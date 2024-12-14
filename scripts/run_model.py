import os
import torch
import time
import itertools
import argparse
from utils.modelTraining import train_and_evaluate

# TODO add runtime & num_trainable_params to the results csv
# TODO Generative model using Flow Matching
print(f"The version of PyTorch is: {torch.__version__}", flush=True)
print("Cuda: ", torch.cuda.is_available(), flush=True)

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run grid search experiments")
parser.add_argument('--batch_index', type=int, default=0, help='Index of the batch to run (starting from 0)')
parser.add_argument('--total_batches', type=int, default=1, help='Total number of batches')
args = parser.parse_args()

# Grid search configuration
grid_search_config = {
    'learning_rate': [0.0001, 0.0005, 0.001],
    'batch_size': [32, 64, 128],
    'depth': [1, 2, 3],
    'hidden_dim': [256, 512],
    'latent_dim': [512, 1024],
    'dropout_prob': [0.1, 0.3],
    'kl_weight': [0.001, 0.01, 0.1],  # Only used for variational models
}

CONFIGS = {
    'single_view_fp_plain': {
        'inputs_0': ['0/Drug_fp'],
        'inputs_1': ['1/Target_fp'],
        'model_type': 'plain',
    },
    'single_view_emb_plain': {
        'inputs_0': ['0/Drug_emb_graph'],
        'inputs_1': ['1/Target_emb_T5'],
        'model_type': 'plain',
    },
    'single_view_fp_variational': {
        'inputs_0': ['0/Drug_fp'],
        'inputs_1': ['1/Target_fp'],
        'model_type': 'variational',
    },
    'single_view_emb_variational': {
        'inputs_0': ['0/Drug_emb_graph'],
        'inputs_1': ['1/Target_emb_T5'],
        'model_type': 'variational',
    },
    'multi_view_plain': {
        'inputs_0': ['0/Drug_fp', '0/Drug_emb_graph', '0/Drug_image', '0/Drug_text'],
        'inputs_1': ['1/Target_fp', '1/Target_emb_T5', '1/Target_emb_DNA', '1/Target_emb_ESM'],
        'model_type': 'plain',
    },
    'multi_view_variational': {
        'inputs_0': ['0/Drug_fp', '0/Drug_emb_graph', '0/Drug_image', '0/Drug_text'],
        'inputs_1': ['1/Target_fp', '1/Target_emb_T5', '1/Target_emb_DNA', '1/Target_emb_ESM'],
        'model_type': 'variational',
    },
}

def generate_experiments(configs, grid_search_config):
    experiments = []
    for config_name, base_config in configs.items():
        model_type = base_config['model_type']
        if model_type == 'plain':
            hyperparams_dict = {k: v for k, v in grid_search_config.items() if k != 'kl_weight'}
        else:
            hyperparams_dict = grid_search_config
        
        keys, values = zip(*hyperparams_dict.items())
        grid_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        
        for hyperparams in grid_combinations:
            full_config = base_config.copy()
            full_config.update(hyperparams)
            experiments.append((config_name, full_config, hyperparams))
    return experiments

def perform_grid_search(batch_experiments):
    for (config_name, full_config, hyperparams) in batch_experiments:
        torch.cuda.empty_cache()
        
        print(f"\nRunning Experiment: {config_name}, Params: {hyperparams}", flush=True)
        t = time.time()
        
        try:
            best_valid_loss, test_loss = train_and_evaluate(
                config=full_config,
                split_type="split_cold",
                **hyperparams
            )
            
            result = {
                'experiment': config_name,
                **hyperparams,
                'best_valid_loss': best_valid_loss,
                'test_loss': test_loss
            }

            # Save results
            results_file = "grid_search_results_split_cold.csv"
            if not os.path.exists(results_file):
                with open(results_file, "w", encoding="utf-8") as f:
                    headers = ["experiment", "learning_rate", "batch_size", "depth", "hidden_dim", "latent_dim", "dropout_prob", "kl_weight", "best_valid_loss", "test_loss"]
                    f.write(",".join(headers) + "\n")

            with open(results_file, "a", encoding="utf-8") as f:
                row = [str(result.get(key, '')) for key in headers]
                f.write(",".join(row) + "\n")
            print("Elapsed time: ", time.time() - t, flush=True)
            
        except Exception as e:
            print(f"Error in experiment {config_name}: {e}" , flush=True)

# Generate all experiments
experiments = generate_experiments(CONFIGS, grid_search_config)
total_experiments = len(experiments) # +-5k
print(f"Total experiments: {total_experiments}", flush=True)

# Split experiments into batches
batch_size = total_experiments // args.total_batches
remainder = total_experiments % args.total_batches

start_idx = args.batch_index * batch_size + min(args.batch_index, remainder)
end_idx = start_idx + batch_size + (1 if args.batch_index < remainder else 0)

batch_experiments = experiments[start_idx:end_idx]
print(f"Running experiments {start_idx} to {end_idx - 1}", flush=True)

# Perform grid search on the selected batch
perform_grid_search(batch_experiments)
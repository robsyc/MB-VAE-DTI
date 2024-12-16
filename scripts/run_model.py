import os
import random
import torch
import time
import itertools
import argparse
import json
from utils.modelTraining import train_and_evaluate

print(f"The version of PyTorch is: {torch.__version__}", flush=True)
print("Cuda: ", torch.cuda.is_available(), flush=True)

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run grid search experiments")
parser.add_argument('--batch_index', type=int, default=0, help='Index of the batch to run (starting from 0)')
parser.add_argument('--total_batches', type=int, default=1, help='Total number of batches')
args = parser.parse_args()

CONFIGS = {
    'single_view_fp_plain': {
        'inputs_0': ['0/Drug_fp'],
        'inputs_1': ['1/Target_fp'],
        'model_type': 'plain',
        'hyperparamter_grid': {
            'learning_rate': [0.0001, 0.0005, 0.001],
            'batch_size': [32, 64, 128],
            'depth': [1, 2, 3, 4],
            'hidden_dim': [64, 128, 256, 512],
            'latent_dim': [128, 256, 512, 1024],
            'dropout_prob': [0.2],
        } # 576 combinations, ~110 secs each = 18 hours
    },
    'single_view_emb_plain': {
        'inputs_0': ['0/Drug_emb_graph'],
        'inputs_1': ['1/Target_emb_T5'],
        'model_type': 'plain',
        'hyperparamter_grid': {
            'learning_rate': [0.0001, 0.0005, 0.001],
            'batch_size': [32, 64, 128],
            'depth': [2, 3, 4],
            'hidden_dim': [64, 128, 256],
            'latent_dim': [256, 512, 1024],
            'dropout_prob': [0.2, 0.4],
        } # 486 combinations, ~90 secs each = 13 hours
    },
    'single_view_fp_variational': {
        'inputs_0': ['0/Drug_fp'],
        'inputs_1': ['1/Target_fp'],
        'model_type': 'variational',
        'hyperparamter_grid': {
            'learning_rate': [0.00005, 0.0001, 0.0005],
            'batch_size': [16, 32, 64, 128],
            'depth': [1, 2, 3],
            'hidden_dim': [64, 128, 256, 512],
            'latent_dim': [128, 256, 512],
            'dropout_prob': [0.2],
            'kl_weight': [0.0001, 0.001, 0.01],
        } # 1296 combinations, ~200 secs each = 72 hours
    },
    'single_view_emb_variational': {
        'inputs_0': ['0/Drug_emb_graph'],
        'inputs_1': ['1/Target_emb_T5'],
        'model_type': 'variational',
        'hyperparamter_grid': {
            'learning_rate': [0.00005, 0.0001, 0.0005],
            'batch_size': [16, 32, 64, 128],
            'depth': [0, 1, 2, 3],
            'hidden_dim': [64, 128, 256],
            'latent_dim': [128, 256, 512],
            'dropout_prob': [0.2],
            "kl_weight": [0.0001, 0.001, 0.01],
        } # 1296 combinations, ~170 secs each = 62 hours
    },
    'multi_view_plain': {
        'inputs_0': ['0/Drug_fp', '0/Drug_emb_graph', '0/Drug_emb_image', '0/Drug_emb_text'],
        'inputs_1': ['1/Target_fp', '1/Target_emb_T5', '1/Target_emb_DNA', '1/Target_emb_ESM'],
        'model_type': 'plain',
        'hyperparamter_grid': {
            'learning_rate': [0.0001, 0.0005, 0.001],
            'batch_size': [64, 128, 256],
            'depth': [1, 2, 3],
            'hidden_dim': [64, 128, 256],
            'latent_dim': [256, 512, 1024],
            'dropout_prob': [0.2],
        } # 243 combinations, ~160 secs each = 11 hours
    },
    'multi_view_variational': {
        'inputs_0': ['0/Drug_fp', '0/Drug_emb_graph', '0/Drug_emb_image', '0/Drug_emb_text'],
        'inputs_1': ['1/Target_fp', '1/Target_emb_T5', '1/Target_emb_DNA', '1/Target_emb_ESM'],
        'model_type': 'variational',
        'hyperparamter_grid': {
            'learning_rate': [0.00005, 0.0001, 0.0005],
            'batch_size': [16, 32, 64, 128],
            'depth': [1, 2, 3],
            'hidden_dim': [64, 128, 256],
            'latent_dim': [128, 256, 512],
            'dropout_prob': [0.2],
            "kl_weight": [0.0001, 0.001, 0.01],
        } # 972 combinations, ~330 secs each = 90 hours
    }, # total 4869 combinations, 266 hours (~11 days)
}

def generate_experiments(configs) -> list:
    experiments = []
    for config_name, config in configs.items():
        hyperparams_dict = config['hyperparamter_grid']
        
        keys, values = zip(*hyperparams_dict.items())
        grid_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        print(f"Config: {config_name}, Grid combinations: {len(grid_combinations)}", flush=True)
        
        for hyperparams in grid_combinations:
            full_config = config.copy()
            full_config.update(hyperparams)
            experiments.append((config_name, full_config, hyperparams))
    
    # shuffle experiments to distribute the load in a reproducible way
    random.seed(42)
    random.shuffle(experiments)
    return experiments

def perform_grid_search(batch_experiments):
    for (config_name, full_config, hyperparams) in batch_experiments:
        torch.cuda.empty_cache()
        
        print(f"\nRunning Experiment: {config_name}, Params: {hyperparams}", flush=True)
        t = time.time()
        
        try:
            best_valid_loss, avg_test_loss, num_trainable_params, predictions = train_and_evaluate(
                config=full_config,
                split_type="split_cold",
                **hyperparams
            )
            elapsed_time = time.time() - t
            experiment_id = f"{config_name}_{hash(frozenset(hyperparams.items()))}"
            result = {
                'experiment': experiment_id,
                **hyperparams,
                'best_valid_loss': best_valid_loss,
                'test_loss': avg_test_loss,
                'num_trainable_params': num_trainable_params,
                'runtime': elapsed_time
            }
            
            # Save predictions to a separate file
            predictions_file = f"./predictions/{experiment_id}.csv"
            os.makedirs(os.path.dirname(predictions_file), exist_ok=True)
            with open(predictions_file, "w", encoding="utf-8") as f:
                for pred in predictions:
                    f.write(f"{pred[0]},{pred[1]}\n")
            
            # Save result to a separate file
            results_file = f"./results/{experiment_id}.json"
            os.makedirs(os.path.dirname(results_file), exist_ok=True)
            with open(results_file, "w", encoding="utf-8") as f:
                json.dump(result, f)
    
        except Exception as e:
            print(f"Error in experiment {config_name}: {e}" , flush=True)
            # Handle exceptions and save error information
            elapsed_time = time.time() - t
            experiment_id = f"{config_name}_{hash(frozenset(hyperparams.items()))}"
            result = {
                'experiment': experiment_id,
                **hyperparams,
                'best_valid_loss': None,
                'test_loss': None,
                'num_trainable_params': None,
                'runtime': elapsed_time,
                'error': str(e)
            }
            # Save the error result to a file
            results_file = f"./results/{experiment_id}.json"
            os.makedirs(os.path.dirname(results_file), exist_ok=True)
            with open(results_file, "w", encoding="utf-8") as f:
                json.dump(result, f)

        print(f"Elapsed time: {elapsed_time:.2f} seconds", flush=True)

# Generate all experiments
experiments = generate_experiments(CONFIGS)
total_experiments = len(experiments) # +- 5k
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
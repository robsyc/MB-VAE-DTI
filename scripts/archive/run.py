import os
import random
import torch
import time
import itertools
import argparse
import json
from utils.training import train_and_evaluate

SEED = 42

print(f"The version of PyTorch is: {torch.__version__}", flush=True)
print("Cuda: ", torch.cuda.is_available(), flush=True)

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run grid search experiments")
parser.add_argument('--batch_index', type=int, default=0, help='Index of the batch to run (starting from 0)')
parser.add_argument('--total_batches', type=int, default=1, help='Total number of batches')
args = parser.parse_args()

CONFIGS = {
    # 'single_view_fp_plain_random_split': {
    #     'inputs_0': ['0/Drug_fp'],
    #     'inputs_1': ['1/Target_fp'],
    #     'model_type': 'plain',
    #     'split_type': 'split_rand',
    #     'hyperparamter_grid': {
    #         'learning_rate': [0.0001, 0.0005, 0.001],
    #         'batch_size': [32, 64, 128],
    #         'depth': [1, 2],
    #         'hidden_dim': [128, 256],
    #         'latent_dim': [256, 512],
    #         'dropout_prob': [0.1, 0.3]
    #     }
    # },
    'single_view_fp_plain_cold_split': {
        'inputs_0': ['0/Drug_fp'],
        'inputs_1': ['1/Target_fp'],
        'model_type': 'plain',
        'split_type': 'split_cold',
        'hyperparamter_grid': {
            'learning_rate': [0.0001, 0.0005, 0.001],
            'batch_size': [32, 64, 128],
            'depth': [1, 2],
            'hidden_dim': [128, 256],
            'latent_dim': [256, 512],
            'dropout_prob': [0.1, 0.3]
        }
    },
    # 'multi_view_plain_random_split': {
    #     'inputs_0': ['0/Drug_emb_graph', '0/Drug_emb_image', '0/Drug_emb_text'],
    #     'inputs_1': ['1/Target_emb_T5', '1/Target_emb_DNA', '1/Target_emb_ESM'],
    #     'model_type': 'plain',
    #     'split_type': 'split_rand',
    #     'hyperparamter_grid': {
    #         'learning_rate': [0.0001, 0.0005, 0.001],
    #         'batch_size': [32, 64, 128],
    #         'depth': [1, 2],
    #         'hidden_dim': [128, 256],
    #         'latent_dim': [256, 512],
    #         'dropout_prob': [0.1, 0.3]
    #     }
    # },
    'multi_view_plain_cold_split': {
        'inputs_0': ['0/Drug_emb_graph', '0/Drug_emb_image', '0/Drug_emb_text'],
        'inputs_1': ['1/Target_emb_T5', '1/Target_emb_DNA', '1/Target_emb_ESM'],
        'model_type': 'plain',
        'split_type': 'split_cold',
        'hyperparamter_grid': {
            'learning_rate': [0.0001, 0.0005, 0.001],
            'batch_size': [32, 64, 128],
            'depth': [1, 2],
            'hidden_dim': [128, 256],
            'latent_dim': [256, 512],
            'dropout_prob': [0.1, 0.3]
        }
    },
    # 'single_view_fp_variational_random_split': {
    #     'inputs_0': ['0/Drug_fp'],
    #     'inputs_1': ['1/Target_fp'],
    #     'model_type': 'variational',
    #     'split_type': 'split_rand',
    #     'hyperparamter_grid': {
    #         'learning_rate': [0.0001, 0.0005, 0.001],
    #         'batch_size': [16, 32, 64],
    #         'depth': [1, 2],
    #         'hidden_dim': [128, 256],
    #         'latent_dim': [256, 512],
    #         'dropout_prob': [0.2],
    #         'kl_weight': [0.0001, 0.001, 0.005],
    #     }
    # },
    'single_view_fp_variational_cold_split': {
        'inputs_0': ['0/Drug_fp'],
        'inputs_1': ['1/Target_fp'],
        'model_type': 'variational',
        'split_type': 'split_cold',
        'hyperparamter_grid': {
            'learning_rate': [0.0001, 0.0005, 0.001],
            'batch_size': [16, 32, 64],
            'depth': [1, 2],
            'hidden_dim': [128, 256],
            'latent_dim': [256, 512],
            'dropout_prob': [0.2],
            'kl_weight': [0.0001, 0.001, 0.005],
        }
    },
    'multi_view_variational_random_split': {
        'inputs_0': ['0/Drug_emb_graph', '0/Drug_emb_image', '0/Drug_emb_text'],
        'inputs_1': ['1/Target_emb_T5', '1/Target_emb_DNA', '1/Target_emb_ESM'],
        'model_type': 'variational',
        'split_type': 'split_rand',
        'hyperparamter_grid': {
            'learning_rate': [0.0001, 0.0005, 0.001],
            'batch_size': [16, 32, 64],
            'depth': [1, 2],
            'hidden_dim': [128, 256],
            'latent_dim': [256, 512],
            'dropout_prob': [0.2],
            'kl_weight': [0.0001, 0.001, 0.005],
        }
    }, 
    'multi_view_variational_cold_split': {
        'inputs_0': ['0/Drug_emb_graph', '0/Drug_emb_image', '0/Drug_emb_text'],
        'inputs_1': ['1/Target_emb_T5', '1/Target_emb_DNA', '1/Target_emb_ESM'],
        'model_type': 'variational',
        'split_type': 'split_cold',
        'hyperparamter_grid': {
            'learning_rate': [0.0001, 0.0005, 0.001],
            'batch_size': [16, 32, 64],
            'depth': [1, 2],
            'hidden_dim': [128, 256],
            'latent_dim': [256, 512],
            'dropout_prob': [0.2],
            'kl_weight': [0.0001, 0.001, 0.005],
        }
    }, 
    # total 1k combinations spread over 12 jobs
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
            del full_config['hyperparamter_grid']
            full_config["hyperparams"] = hyperparams
            experiments.append((config_name, full_config))
    
    # shuffle experiments to distribute the load in a reproducible way
    random.seed(SEED)
    random.shuffle(experiments)
    return experiments

def perform_grid_search(batch_experiments):
    for (config_name, config) in batch_experiments:
        torch.cuda.empty_cache()
        
        print(f"\nRunning Experiment: {config_name}, Hyperparams: {config['hyperparams']}", flush=True)
        t = time.time()
        
        try:
            best_valid_loss, avg_test_loss, num_trainable_params, predictions = train_and_evaluate(config=config)
            elapsed_time = time.time() - t
            experiment_id = f"{config_name}_{hash(frozenset(config['hyperparams'].items()))}"
            result = {
                'experiment': experiment_id,
                **config['hyperparams'],
                'best_valid_loss': best_valid_loss,
                'test_loss': avg_test_loss,
                'num_trainable_params': num_trainable_params,
                'runtime': elapsed_time
            }
            
            # Save predictions to a separate file
            # predictions_file = f"./predictions/{experiment_id}.csv"
            # os.makedirs(os.path.dirname(predictions_file), exist_ok=True)
            # with open(predictions_file, "w", encoding="utf-8") as f:
            #     for pred in predictions:
            #         f.write(f"{pred[0]},{pred[1]}\n")
            
            # Save result to a separate file
            results_file = f"./results/{experiment_id}.json"
            os.makedirs(os.path.dirname(results_file), exist_ok=True)
            with open(results_file, "w", encoding="utf-8") as f:
                json.dump(result, f)
    
        except Exception as e:
            print(f"Error in experiment {config_name}: {e}" , flush=True)
            elapsed_time = time.time() - t

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
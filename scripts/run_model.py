import os
import sys
import torch
import time
import itertools

# os.chdir('/home/robbec/thesis/MB-VAE-DTI/')
# sys.path.append('/home/robbec/thesis/MB-VAE-DTI/')
os.chdir('/home/robsyc/Desktop/thesis/MB-VAE-DTI/')
sys.path.append('/home/robsyc/Desktop/thesis/MB-VAE-DTI/')
print(os.listdir('.'))
print(os.getcwd())

print(torch.cuda.is_available())
torch.cuda.empty_cache()

from utils.modelTraining import train_and_evaluate

grid_search_config = {
    'learning_rate': [0.0001, 0.0005, 0.001, 0.01],
    'batch_size': [16, 32, 64, 128],
    'depth': [0, 1, 2, 3],
    'hidden_dim': [128, 256, 512],
    'latent_dim': [256, 512, 1024],
    'dropout_prob': [0.1, 0.3, 0.5],
    # 'kl_weight': [0.01, 0.1, 0.5],  # Only used for variational models
}

CONFIGS = {
    'single_view_fp_plain': {
        'inputs_0': ['0/Drug_fp'],
        'inputs_1': ['1/Target_fp'],
        'model_type': 'plain',  # 'plain' or 'variational'
    },
    # 'single_view_emb_plain': {
    #     'inputs_0': ['0/Drug_emb_graph'],
    #     'inputs_1': ['1/Target_emb_T5'],
    #     'model_type': 'plain',
    # }
}

def perform_grid_search(configs, grid_search_config):
    print("Starting Grid Search...")
    results = []
    
    # Generate all possible combinations
    keys, values = zip(*grid_search_config.items())
    grid_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    for config_name, base_config in configs.items():
        for hyperparams in grid_combinations:
            
            # Merge base config with hyperparameters
            full_config = base_config.copy()
            full_config.update(hyperparams)
            
            print(f"\nRunning Grid Search: {config_name}, Params: {hyperparams}")
            
            try:
                best_valid_loss, test_loss = train_and_evaluate(
                    config=full_config,
                    split_type="split_rand",
                    num_epochs=30,
                    **hyperparams
                )
                
                result = {
                    'experiment': config_name,
                    **hyperparams,
                    'best_valid_loss': best_valid_loss,
                    'test_loss': test_loss
                }
                results.append(result)
                
            except Exception as e:
                print(f"Error in experiment {config_name}: {e}")
    
    return results

# Perform grid search
grid_search_results = perform_grid_search(CONFIGS, grid_search_config)

# Optional: Save results to CSV for further analysis
import pandas as pd
results_df = pd.DataFrame(grid_search_results)
results_df.to_csv('grid_search_results_simple_non_var.csv', index=False)
import os
import torch
import time
import itertools
from utils.modelTraining import train_and_evaluate

print(os.getcwd())
print(os.listdir('.'))

print(f"The version of PyTorch is: {torch.__version__}")
print("Cuda: ", torch.cuda.is_available())
torch.cuda.empty_cache()

print("Import successful")

grid_search_config = {
    # 'learning_rate': [0.0001, 0.0005, 0.001, 0.01],  
    # 'batch_size': [16, 32, 64, 128],
    # 'depth': [0, 1, 2, 3],
    # 'hidden_dim': [128, 256, 512],
    # 'latent_dim': [256, 512, 1024],
    # 'dropout_prob': [0.1, 0.3],
    # 'kl_weight': [0.001, 0.01, 0.1, 1.],  # Only used for variational models
    'learning_rate': [0.0001],
    'batch_size': [64],
    'depth': [3],
    'hidden_dim': [512],
    'latent_dim': [1024],
    'dropout_prob': [0.3],
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
    
    # Generate all possible combinations
    keys, values = zip(*grid_search_config.items())
    grid_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    for config_name, base_config in configs.items():
        for hyperparams in grid_combinations:
            torch.cuda.empty_cache()
            
            # Merge base config with hyperparameters
            full_config = base_config.copy()
            full_config.update(hyperparams)
            
            print(f"\nRunning Grid Search: {config_name}, Params: {hyperparams}")
            t = time.time()
            
            try:
                best_valid_loss, test_loss = train_and_evaluate(
                    config=full_config,
                    split_type="split_rand",
                    **hyperparams
                )
                
                result = {
                    'experiment': config_name,
                    **hyperparams,
                    'best_valid_loss': best_valid_loss,
                    'test_loss': test_loss
                }

                # Save results
                if not os.path.exists("grid_search_results_split_rand.csv"):
                    with open("grid_search_results.csv", "w", encoding="utf-8") as f:
                        f.write("experiment,learning_rate,batch_size,depth,hidden_dim,latent_dim,dropout_prob,best_valid_loss,test_loss\n")

                with open("grid_search_results_split_rand.csv", "a", encoding="utf-8") as f:
                    f.write(",".join(str(result[key]) for key in ["experiment", "learning_rate", "batch_size", "depth", "hidden_dim", "latent_dim", "dropout_prob", "best_valid_loss", "test_loss"]))
                    f.write("\n")
                print("Elapsed time: ", time.time() - t)
                
            except Exception as e:
                print(f"Error in experiment {config_name}: {e}")

# Perform grid search
perform_grid_search(CONFIGS, grid_search_config)
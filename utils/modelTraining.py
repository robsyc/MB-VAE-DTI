import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import h5torch
import copy
from utils.modelBuilding import (
    PlainMultiBranch,
    VariationalMultiBranch
)

# Configurations for different experiments
CONFIGS = {
    'single_view_fp': {
        'inputs_0': ['0/Drug_fp'],
        'inputs_1': ['1/Target_fp'],
        'model_type': 'plain',  # 'plain' or 'variational'
    },
    # 'single_view_emb': {
    #     'inputs_0': ['0/Drug_emb_graph'],
    #     'inputs_1': ['1/Target_emb_T5'],
    #     'model_type': 'plain',
    # },
    # 'var_single_view_fp': {
    #     'inputs_0': ['0/Drug_fp'],
    #     'inputs_1': ['1/Target_fp'],
    #     'model_type': 'variational',
    # },
    # 'var_single_view_emb': {
    #     'inputs_0': ['0/Drug_emb_graph'],
    #     'inputs_1': ['1/Target_emb_T5'],
    #     'model_type': 'variational',
    # },
    # 'multi_view': {
    #     'inputs_0': ['0/Drug_fp', '0/Drug_emb_graph', '0/Drug_emb_image', '0/Drug_emb_text'],
    #     'inputs_1': ['1/Target_fp', '1/Target_emb_ESM', '1/Target_emb_T5', '1/Target_emb_DNA'],
    #     'model_type': 'variational',
    # },
    # Add more configurations if needed
}

def get_dataset(split_type, split_name):
    return h5torch.Dataset(
        "./data/dataset/DAVIS.h5t",
        sampling="coo",
        subset=(f"unstructured/{split_type}", split_name),
        in_memory=True,
    )

class CustomH5Dataset(Dataset):
    def __init__(self, h5_dataset, inputs_0, inputs_1):
        self.dataset = h5_dataset
        self.inputs_0 = inputs_0
        self.inputs_1 = inputs_1

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        batch = self.dataset[idx]
        # Inputs for branch 0 (drug)
        x0 = [batch[key] for key in self.inputs_0]
        # Inputs for branch 1 (target)
        x1 = [batch[key] for key in self.inputs_1]
        # Interaction value
        y = batch['central']
        return x0, x1, y

# Training and evaluation function
def train_and_evaluate(config, split_type, num_epochs=10, batch_size=32):
    # Prepare datasets
    train_dataset = get_dataset(split_type, 'train')
    valid_dataset = get_dataset(split_type, 'valid')
    test_dataset = get_dataset(split_type, 'test')

    # Create DataLoaders
    train_loader = DataLoader(
        CustomH5Dataset(train_dataset, config['inputs_0'], config['inputs_1']),
        batch_size=batch_size,
        shuffle=True
    )
    valid_loader = DataLoader(
        CustomH5Dataset(valid_dataset, config['inputs_0'], config['inputs_1']),
        batch_size=batch_size,
        shuffle=False
    )
    test_loader = DataLoader(
        CustomH5Dataset(test_dataset, config['inputs_0'], config['inputs_1']),
        batch_size=batch_size,
        shuffle=False
    )

    # Determine input dimensions
    sample_batch = next(iter(train_loader))
    input_dim_list_0 = [x.shape[1] for x in sample_batch[0]]
    input_dim_list_1 = [x.shape[1] for x in sample_batch[1]]

    # Define the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config['model_type'] == 'plain':
        model = PlainMultiBranch(
            input_dim_list_0=input_dim_list_0,
            input_dim_list_1=input_dim_list_1,
            hidden_dim=512,
            latent_dim=1024,
            depth=1,
            dropout_prob=0.1
        ).to(device)
    else:
        model = VariationalMultiBranch(
            input_dim_list_0=input_dim_list_0,
            input_dim_list_1=input_dim_list_1,
            hidden_dim=512,
            latent_dim=1024,
            depth=1,
            dropout_prob=0.1
        ).to(device)

    # Loss function and optimizer
    mse_loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    kl_weight = 1e-4  # Adjust KL divergence weight if using variational models

    # Variables to track the best model
    best_valid_loss = float('inf')
    best_model_state = None

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for x0, x1, y in train_loader:
            # Move data to device
            x0 = [x.to(device).float() for x in x0]
            x1 = [x.to(device).float() for x in x1]
            y = y.to(device).float()

            # Forward pass
            if config['model_type'] == 'plain':
                output, coeffs = model(x0, x1)
                loss = mse_loss_fn(output.squeeze(), y)
            else:
                output, kl_loss, coeffs = model(x0, x1, compute_loss=True)
                mse_loss = mse_loss_fn(output.squeeze(), y)
                loss = mse_loss + kl_weight * kl_loss

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation
        model.eval()
        total_valid_loss = 0
        with torch.no_grad():
            for x0, x1, y in valid_loader:
                # Move data to device
                x0 = [x.to(device).float() for x in x0]
                x1 = [x.to(device).float() for x in x1]
                y = y.to(device).float()

                # Forward pass
                if config['model_type'] == 'plain':
                    output, coeffs = model(x0, x1)
                    loss = mse_loss_fn(output.squeeze(), y)
                else:
                    output, kl_loss = model(x0, x1, compute_loss=False)
                    loss = mse_loss_fn(output.squeeze(), y)

                total_valid_loss += loss.item()

        avg_valid_loss = total_valid_loss / len(valid_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Valid Loss: {avg_valid_loss:.4f}")

        # Save the best model
        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            best_model_state = copy.deepcopy(model.state_dict())

    # Load the best model
    model.load_state_dict(best_model_state)

    # Test the model
    model.eval()
    total_test_loss = 0
    with torch.no_grad():
        for x0, x1, y in test_loader:
            # Move data to device
            x0 = [x.to(device).float() for x in x0]
            x1 = [x.to(device).float() for x in x1]
            y = y.to(device).float()

            # Forward pass
            if config['model_type'] == 'plain':
                output, coeffs = model(x0, x1)
                loss = mse_loss_fn(output.squeeze(), y)
            else:
                output, kl_loss = model(x0, x1, compute_loss=False)
                loss = mse_loss_fn(output.squeeze(), y)

            total_test_loss += loss.item()

    avg_test_loss = total_test_loss / len(test_loader)
    print(f"Test Loss: {avg_test_loss:.4f}")

    # Return the best validation loss and corresponding test loss
    return best_valid_loss, avg_test_loss

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import h5torch
import copy
from utils.modelBuilding import MultiBranchDTI


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

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Training and evaluation function
def train_and_evaluate(
        config, 
        split_type, 
        learning_rate,
        batch_size,
        hidden_dim,
        latent_dim,
        depth,
        kl_weight=None,
        dropout_prob=0.1,
        output_setting="regression" # TDOO add support for classification
    ):
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
    model = MultiBranchDTI(
        input_dim_list_0=input_dim_list_0,
        input_dim_list_1=input_dim_list_1,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        depth=depth,
        dropout_prob=dropout_prob,
        variational=True if config['model_type'] == 'variational' else False
    ).to(device)

    # Loss function and optimizer
    mse_loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Variables to track the best model & early stopping
    epoch = 0
    patience = 12
    early_stopping_counter = 0
    best_valid_loss = float('inf')
    best_model_state = copy.deepcopy(model.state_dict())

    def move_to_device(data, device):
        data = [x.to(device).float() for x in data]
        return data[0] if len(data) == 1 else data  # Return single tensor if only one input (single view case)
    
    # Training loop
    while True:
        epoch += 1
        model.train()
        total_train_loss = 0
        for x0, x1, y in train_loader:
            x0 = move_to_device(x0, device)
            x1 = move_to_device(x1, device)
            y = y.to(device).float()

            # Forward pass
            if config['model_type'] == 'plain':
                output = model(x0, x1)
                loss = mse_loss_fn(output.squeeze(), y)
            else:
                output, kl_loss = model(x0, x1, compute_kl_loss=True)
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
                x0 = move_to_device(x0, device)
                x1 = move_to_device(x1, device)
                y = y.to(device).float()

                # Forward pass (no kl_loss computation)
                output = model(x0, x1)
                loss = mse_loss_fn(output.squeeze(), y)

                total_valid_loss += loss.item()

        avg_valid_loss = total_valid_loss / len(valid_loader)
        print(f"Epoch [{epoch}], Train Loss: {avg_train_loss:.4f}, Valid Loss: {avg_valid_loss:.4f}")

        # Save the best model
        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            best_model_state = copy.deepcopy(model.state_dict())
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
        
        # Early stopping
        if early_stopping_counter >= patience:
            break

    # Load the best model
    print(f"Best Validation Loss: {best_valid_loss:.4f}")
    model.load_state_dict(best_model_state)
    num_trainable_params = count_parameters(model)

    # Test the model
    model.eval()
    total_test_loss = 0
    predictions = []
    with torch.no_grad():
        for x0, x1, y in test_loader:
            # Move data to device
            x0 = move_to_device(x0, device)
            x1 = move_to_device(x1, device)
            y = y.to(device).float()

            # Forward pass (no kl_loss computation)
            output = model(x0, x1)
            loss = mse_loss_fn(output.squeeze(), y)            
            predictions.append((output.squeeze().cpu().numpy(), y.cpu().numpy()))
            total_test_loss += loss.item()

    avg_test_loss = total_test_loss / len(test_loader)
    # TODO: plot residuals, other metrics e.g. concordance index
    print(f"Test Loss: {avg_test_loss:.4f}")

    # Return the best validation loss and corresponding test loss
    return best_valid_loss, avg_test_loss, num_trainable_params, predictions
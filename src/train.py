import os
import yaml
import torch
from src.data_setup import create_dataloader
from src.model_builder import SteganoModel
from src.loss import SteganoLoss
from src.engine import train

def main():
    # 1. Load configuration
    config_path = "configs/config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # 2. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 3. Create Artifacts directory
    os.makedirs(os.path.dirname(config['model_path']), exist_ok=True)

    # 4. Initialize Data Loaders
    train_loader = create_dataloader(config, mode='train')
    val_loader = create_dataloader(config, mode='val')

    # 5. Initialize Model
    model = SteganoModel().to(device)

    # 6. Initialize Loss (Criterion)
    criterion = SteganoLoss(
        edge_alpha=config['edge_alpha'],
        secret_weight=config['secret_weight'],
        sparsity_weight=config['sparsity_weight']
    )

    # 7. Start Training
    print("Starting Training Pipeline...")
    train(model, train_loader, val_loader, criterion, config, device)

if __name__ == "__main__":
    main()
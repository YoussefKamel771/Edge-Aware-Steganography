import yaml
from src.data_setup import create_dataloader
from src.model_builder import SteganoModel
import torch


def main():

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load the configuration from the file
    try:
        with open("configs/config.yaml", "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print("Error: config.yaml not found in configs/ directory.")
        return

    # 2. Pass the config object into the dataloader factory
    train_loader = create_dataloader(config, mode='train')
    val_loader = create_dataloader(config, mode='val')


    model = SteganoModel().to(DEVICE)
    # print(model)

    # 3. Start training
    # train_model(train_loader, val_loader, config)

if __name__ == "__main__":
    main()
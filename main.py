import yaml
from src.data_setup import create_dataloader

def main():
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

    print(f"Loaded {len(train_loader)} training batches.")
    print(f"Loaded {len(val_loader)} validation batches.")

    # 3. Start training
    # train_model(train_loader, val_loader, config)

if __name__ == "__main__":
    main()
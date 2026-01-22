import torch
import matplotlib.pyplot as plt
from src.model_builder import SteganoModel


def visualize_results(model, loader, device):
    model.eval()
    with torch.no_grad():
        cover, secret, edge, _ = next(iter(loader))
        cover, secret, edge = cover.to(device), secret.to(device), edge.to(device)

        stego, feat, fused = model.hide_network(cover, secret, edge)
        recovered = model.reveal_network(stego, edge, feat, fused)
        
        diff = (stego - cover).abs().sum(dim=1, keepdim=True)
        diff = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)

        images = [cover[0], secret[0], edge[0].squeeze(), stego[0], recovered[0], diff[0].squeeze()]
        titles = ["Cover", "Secret", "Edge", "Stego", "Recovered", "Diff"]

        plt.figure(figsize=(18, 4))
        for i, img in enumerate(images):
            plt.subplot(1, 6, i+1)
            plt.imshow(img.cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5 if img.dim()==3 else img.cpu().numpy(), 
                       cmap='gray' if i in [2, 5] else None)
            plt.title(titles[i])
            plt.axis('off')
        plt.show()

def load_checkpoint(checkpoint_path, device='cpu'):
    """Initializes model and loads weights from a .pth file."""
    try:
        model = SteganoModel()
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()
        print(f"Successfully loaded checkpoint from {checkpoint_path}")
        return model
    except FileNotFoundError:
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint: {e}")

def plot_stego_results(images, titles, save_path=None):
    """Shared visualization helper for inference results."""
    plt.figure(figsize=(15, 5))
    for i in range(len(titles)):
        plt.subplot(1, len(titles), i + 1)
        # Check if image is grayscale (Edge Map)
        cmap = 'gray' if "Edge" in titles[i] or images[i].ndim == 2 else None
        plt.imshow(images[i], cmap=cmap)
        plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

# Load the model
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model = load_checkpoint(CNN_PATH_NEW, device)
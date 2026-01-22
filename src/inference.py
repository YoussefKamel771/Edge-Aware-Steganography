import torch
import numpy as np
from PIL import Image
from src.utils import plot_stego_results

def inference_with_dataloader(model, dataloader, device):
    """Runs inference on the next batch from a DataLoader."""
    model.eval()
    with torch.no_grad():
        cover, secret, edge, _ = next(iter(dataloader))
        cover, secret, edge = cover.to(device), secret.to(device), edge.to(device)
        stego, recovered = model(cover, secret, edge)

    # Prepare for plotting
    images = [
        cover[0].cpu().permute(1, 2, 0) * 0.5 + 0.5,
        secret[0].cpu().permute(1, 2, 0) * 0.5 + 0.5,
        edge[0].cpu().squeeze(),
        stego[0].cpu().permute(1, 2, 0) * 0.5 + 0.5,
        recovered[0].cpu().permute(1, 2, 0) * 0.5 + 0.5
    ]
    titles = ["Cover", "Secret", "Edge Map", "Stego", "Recovered"]
    plot_stego_results(images, titles)

def inference_with_images(model, cover_path, secret_path, edge_path, 
                          rgb_transform, edge_transform, device):
    """Runs inference on individual image files."""
    model.eval()
    
    # Load and process images
    cover = Image.open(cover_path).convert('RGB')
    secret = Image.open(secret_path).convert('RGB')
    edge = Image.open(edge_path).convert('L')

    seed = np.random.randint(2147483647)
    torch.manual_seed(seed)
    cover_t = rgb_transform(cover).unsqueeze(0).to(device)
    torch.manual_seed(seed)
    secret_t = rgb_transform(secret).unsqueeze(0).to(device)
    torch.manual_seed(seed)
    edge_t = edge_transform(edge).unsqueeze(0).to(device)
    
    # Apply binary threshold
    edge_t = (edge_t > 0.12).float()

    with torch.no_grad():
        stego, recovered = model(cover_t, secret_t, edge_t)

    images = [
        cover_t[0].cpu().permute(1, 2, 0) * 0.5 + 0.5,
        secret_t[0].cpu().permute(1, 2, 0) * 0.5 + 0.5,
        edge_t[0].cpu().squeeze(),
        stego[0].cpu().permute(1, 2, 0) * 0.5 + 0.5,
        recovered[0].cpu().permute(1, 2, 0) * 0.5 + 0.5
    ]
    plot_stego_results(images, ["Cover", "Secret", "Edge Map", "Stego", "Recovered"])
    
    return stego, recovered
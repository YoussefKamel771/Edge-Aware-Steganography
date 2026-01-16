import torch
import matplotlib.pyplot as plt


def visualize(model, loader, device):
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
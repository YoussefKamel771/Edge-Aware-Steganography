import torch
import torch.nn as nn
import torch.nn.functional as F

class SteganoLoss(nn.Module):
    """
    Composite loss function for Edge-Constrained Steganography.
    """
    def __init__(self, edge_alpha=10.0, secret_weight=3.0, sparsity_weight=0.01):
        super(SteganoLoss, self).__init__()
        self.edge_alpha = edge_alpha
        self.secret_weight = secret_weight
        self.sparsity_weight = sparsity_weight

    def edge_constrained_loss(self, stego, cover, edge_map):
        """Calculates loss heavily weighted towards edge regions."""
        non_edge_mask = 1 - edge_map
        # Loss in smooth areas
        content_loss = F.l1_loss(stego * non_edge_mask, cover * non_edge_mask)
        # Loss in texture/edge areas (weighted by alpha)
        edge_loss = F.l1_loss(stego * edge_map, cover * edge_map)
        return content_loss + self.edge_alpha * edge_loss

    def forward(self, stego, cover, recovered, secret, edge_map):
        # 1. Invisibility Loss (how much did we change the cover?)
        l_stego = self.edge_constrained_loss(stego, cover, edge_map)
        
        # 2. Recovery Loss (how well did we get the secret back?)
        l_secret = F.mse_loss(recovered, secret)
        
        # 3. Sparsity Loss (encourages smaller modifications)
        l_sparsity = torch.mean(torch.abs(stego - cover) * edge_map)
        
        total_loss = (0.5 * l_stego) + (self.secret_weight * l_secret) + (self.sparsity_weight * l_sparsity)
        
        return total_loss, {"stego_loss": l_stego.item(), "secret_loss": l_secret.item()}
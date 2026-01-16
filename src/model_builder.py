import torch
import torch.nn as nn
import torch.nn.functional as F

class HideNetwork(nn.Module):
    """
    Network responsible for hiding a secret image within a cover image 
    using an edge-map as a spatial constraint.
    """
    def __init__(self):
        super().__init__()

        # Secret Image Encoder (multi-scale features)
        self.secret_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.GroupNorm(8, 32),  # Replace BatchNorm2d
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GroupNorm(16, 64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GroupNorm(32, 128),
            nn.LeakyReLU(0.2)
        )

        # Edge-Aware Fusion Module
        self.fusion = nn.Sequential(
            nn.Conv2d(128 + 3 + 1, 256, 3, padding=1),  # secret_feat + cover + edge
            nn.GroupNorm(32, 256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.GroupNorm(32, 128),
            nn.LeakyReLU(0.2)
        )

        # Edge-Constrained Decoder (no upsampling, fixed from previous issue)
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.GroupNorm(16, 64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, cover, secret, edge_map):
        # 1. Extract multi-scale secret features
        secret_feat = self.secret_encoder(secret)  # (B, 128, H/4, W/4)
        secret_feat_up = F.interpolate(secret_feat, size=(cover.shape[2], cover.shape[3]), mode='bilinear', align_corners=False)

        # 2. Edge-guided fusion
        x = torch.cat([cover, edge_map, secret_feat_up], dim=1)
        fused = self.fusion(x)

        # 3. Generate edge-constrained modifications
        delta = self.decoder(fused)

        # 4. Strict edge masking
        edge_mask = edge_map.repeat(1, 3, 1, 1)
        stego = cover + delta * edge_mask

        # Return stego and intermediate features for skip connections
        return stego, secret_feat, fused

class RevealNetwork(nn.Module):
    """
    Network responsible for recovering the hidden secret image 
    from the stego image and auxiliary features.
    """
    def __init__(self):
        super().__init__()
        # Input: stego (3 channels) + edge_map (1 channel) + skip connections (128 from secret_feat + 128 from fused)
        self.net = nn.Sequential(
            nn.Conv2d(3 + 1 + 128 + 128, 256, 3, padding=1),  # Combine stego, edge_map, secret_feat, fused
            nn.GroupNorm(32, 256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.GroupNorm(32, 128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.GroupNorm(16, 64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, stego, edge_map, secret_feat, fused):
        # Upsample secret_feat to match stego dimensions
        secret_feat = F.interpolate(secret_feat, size=(stego.shape[2], stego.shape[3]), mode='bilinear', align_corners=False)
        # Concatenate inputs
        x = torch.cat([stego, edge_map, secret_feat, fused], dim=1)
        return self.net(x)

class SteganoModel(nn.Module):
    """
    Wrapper model that combines Hide and Reveal networks for easier management.
    """
    def __init__(self):
        super().__init__()
        self.hide_network = HideNetwork()
        self.reveal_network = RevealNetwork()

    def forward(self, cover, secret, edge_map):
        stego, secret_feat, fused = self.hide_network(cover, secret, edge_map)
        recovered_secret = self.reveal_network(stego, edge_map, secret_feat, fused)
        return stego, recovered_secret
import torch
import torch.nn as nn

class TrajectoryAutoencoder(nn.Module):
    """
    Simple Fully Connected Autoencoder for Trajectories.
    Input: Flattened Trajectory (K * Features)
    """
    def __init__(self, input_dim: int, latent_dim: int = 16):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )
        
    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon

    def get_reconstruction_error(self, x):
        """Returns MSE loss per sample"""
        recon = self.forward(x)
        # MSE per sample (keep batch dimension)
        loss = torch.mean((x - recon) ** 2, dim=1)
        return loss


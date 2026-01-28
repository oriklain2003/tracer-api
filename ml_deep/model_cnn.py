import torch
import torch.nn as nn

class TrajectoryCNN(nn.Module):
    """
    Deep 1D Convolutional Autoencoder.
    Captures local temporal patterns (jitters, turns) better than simple dense networks.
    
    Input Shape: (Batch, Features, TimeSteps) e.g. (32, 4, 50)
    """
    def __init__(self, num_features: int = 4, seq_len: int = 50):
        super().__init__()
        self.seq_len = seq_len
        
        # Encoder
        # Input: [B, 4, 50]
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=num_features, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(2), # [B, 32, 25]
            
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(2), # [B, 64, 12]
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), # [B, 32, 24] -> need to fix output padding to match
            nn.ReLU(),
            nn.BatchNorm1d(32),
            
            nn.ConvTranspose1d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1), # [B, 16, 50]
            nn.ReLU(),
            nn.BatchNorm1d(16),
            
            nn.Conv1d(16, num_features, kernel_size=3, padding=1) # [B, 4, 50]
        )
        
    def forward(self, x):
        # x is [Batch, Seq, Feat], we need [Batch, Feat, Seq] for Conv1d
        x = x.permute(0, 2, 1)
        
        z = self.encoder(x)
        recon = self.decoder(z)
        
        # Permute back to [Batch, Seq, Feat]
        return recon.permute(0, 2, 1)

    def get_reconstruction_error(self, x):
        """Returns MSE loss per sample"""
        recon = self.forward(x)
        # MSE per sample (average over time and features)
        loss = torch.mean((x - recon) ** 2, dim=(1, 2))
        return loss


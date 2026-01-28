import torch
import torch.nn as nn

class TrajectoryCNN(nn.Module):
    """
    Deep 1D Convolutional Autoencoder.
    """
    def __init__(self, num_features: int = 4, seq_len: int = 50):
        super().__init__()
        self.seq_len = seq_len
        
        # Encoder
        self.encoder = nn.Sequential(
            # [B, 4, 50]
            nn.Conv1d(in_channels=num_features, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(2), # -> [B, 32, 25]
            
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(2), # -> [B, 64, 12] (integer division of 25/2 = 12)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            # We need to get back to 25 from 12. 
            # ConvTranspose output = (input - 1)*stride + kernel - 2*padding + output_padding
            # (12-1)*2 + 3 - 2 + 1 = 22 + 2 = 24... close.
            # Let's align manually.
            
            nn.ConvTranspose1d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), # [B, 32, 24] -> if input 12
            # Wait, 12 -> 25? 
            # (12-1)*2 + 3 - 2 + 1 = 24. We need 25.
            # We can use linear interpolation (Upsample) for safety instead of tricky transposed conv math
        )
        
        # Simpler Upsampling Decoder
        self.upsample_decoder = nn.Sequential(
            nn.Upsample(scale_factor=2.0, mode='linear', align_corners=False), # 12->24
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            
            nn.Upsample(size=50, mode='linear', align_corners=False), # Force to 50
            nn.Conv1d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            
            nn.Conv1d(16, num_features, kernel_size=3, padding=1) # [B, 4, 50]
        )
        
    def forward(self, x):
        # x: [Batch, Seq, Feat] -> [Batch, Feat, Seq]
        x = x.permute(0, 2, 1)
        
        z = self.encoder(x)
        recon = self.upsample_decoder(z)
        
        # -> [Batch, Seq, Feat]
        return recon.permute(0, 2, 1)

    def get_reconstruction_error(self, x):
        recon = self.forward(x)
        loss = torch.mean((x - recon) ** 2, dim=(1, 2))
        return loss


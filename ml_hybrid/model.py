import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class HybridAutoencoder(nn.Module):
    """
    Hybrid CNN-Transformer Autoencoder.
    Uses 1D Conv for local feature extraction and Transformer for global temporal dependencies.
    """
    def __init__(self, input_dim=4, seq_len=50, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        
        # --- Encoder ---
        # 1. CNN Feature Extractor
        # Input: [Batch, Features, Seq]
        self.cnn_encoder = nn.Sequential(
            nn.Conv1d(input_dim, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            # We keep sequence length same for simplicity to feed into Transformer
            # Or we could pool. Let's keep it same to preserve temporal resolution for now.
        )
        
        # Projection to d_model
        self.feature_projection = nn.Linear(32, d_model)
        
        # 2. Transformer Encoder
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # --- Decoder ---
        # 3. Transformer Decoder (Mirroring)
        decoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_decoder = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)
        
        # 4. CNN Decoder (Reconstruction)
        self.feature_expansion = nn.Linear(d_model, 32)
        
        self.cnn_decoder = nn.Sequential(
            nn.Conv1d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Conv1d(16, input_dim, kernel_size=3, padding=1)
        )

    def forward(self, x):
        # x: [Batch, Seq, Feat]
        
        # CNN expects [Batch, Feat, Seq]
        x_cnn_in = x.permute(0, 2, 1)
        features = self.cnn_encoder(x_cnn_in) # -> [Batch, 32, Seq]
        
        # Prepare for Transformer: [Batch, Seq, 32]
        features_seq = features.permute(0, 2, 1)
        
        # Project to d_model
        x_trans_in = self.feature_projection(features_seq) # -> [Batch, Seq, d_model]
        x_trans_in = self.pos_encoder(x_trans_in)
        
        # Transformer Encode
        latent = self.transformer_encoder(x_trans_in)
        
        # Transformer Decode
        decoded_latent = self.transformer_decoder(latent)
        
        # Back to CNN features
        decoded_features = self.feature_expansion(decoded_latent) # -> [Batch, Seq, 32]
        
        # CNN Decode
        decoded_features_cnn = decoded_features.permute(0, 2, 1) # -> [Batch, 32, Seq]
        recon = self.cnn_decoder(decoded_features_cnn) # -> [Batch, Feat, Seq]
        
        return recon.permute(0, 2, 1) # -> [Batch, Seq, Feat]

    def get_reconstruction_error(self, x):
        recon = self.forward(x)
        return torch.mean((x - recon) ** 2, dim=(1, 2))

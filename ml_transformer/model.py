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
        # x shape: (batch, seq_len, d_model)
        # pe shape: (1, max_len, d_model)
        return x + self.pe[:, :x.size(1), :]

class TrajectoryTransformerAE(nn.Module):
    def __init__(self, input_dim=4, d_model=64, nhead=4, num_layers=3, dim_feedforward=128, seq_len=50, dropout=0.1):
        super().__init__()
        
        self.input_embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len)
        
        # Encoder Layer
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Bottleneck (optional, to force compression)
        # We project down to d_model/2 and back up
        self.bottleneck_down = nn.Linear(d_model, d_model // 2)
        self.bottleneck_up = nn.Linear(d_model // 2, d_model)
        
        # Decoder Layer (using Encoder structure for simplicity as we have full sequence)
        decoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerEncoder(decoder_layers, num_layers=num_layers)
        
        self.output_projection = nn.Linear(d_model, input_dim)
        
    def forward(self, src):
        # src: (Batch, Seq_Len, Features)
        
        # 1. Embed & Positional Encode
        x = self.input_embedding(src) # (B, S, d_model)
        x = self.pos_encoder(x)
        
        # 2. Encode
        encoded = self.transformer_encoder(x) # (B, S, d_model)
        
        # 3. Bottleneck (Point-wise)
        latent = self.bottleneck_down(encoded) # (B, S, d_model/2)
        latent = torch.relu(latent)
        decoded_latent = self.bottleneck_up(latent) # (B, S, d_model)
        
        # 4. Decode
        # We add pos encoding again to the latent representation before decoding? 
        # Usually not strictly necessary if structure is preserved, but good practice.
        decoded_latent = self.pos_encoder(decoded_latent)
        reconstructed_features = self.transformer_decoder(decoded_latent) # (B, S, d_model)
        
        # 5. Project to Output
        output = self.output_projection(reconstructed_features) # (B, S, Features)
        
        return output

    def get_reconstruction_error(self, x):
        recon = self.forward(x)
        # MSE per sample: Mean over seq_len and features
        error = torch.mean((x - recon) ** 2, dim=(1, 2))
        return error


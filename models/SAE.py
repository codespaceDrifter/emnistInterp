# sparse autoencoder for mechanistic interpretability
# learns an overcomplete sparse dictionary of features from hidden layer activations
# encoder: Linear + ReLU (sparse coding)
# decoder: Linear, no bias, unit-norm columns (feature directions in activation space)
# loss: MSE reconstruction + l1_coeff * L1(encoded)

import torch
import torch.nn as nn


class SAE(nn.Module):
    def __init__(self, input_dim, dict_size, l1_coeff=1e-3):
        super().__init__()
        # input_dim: hidden layer dimension (e.g. 192)
        # dict_size: overcomplete dictionary size (e.g. 192 * 4 = 768)
        self.l1_coeff = l1_coeff

        # (input_dim,) -> (dict_size,) sparse feature activations
        self.encoder = nn.Linear(input_dim, dict_size)
        # (dict_size,) -> (input_dim,) reconstruction, no bias
        self.decoder = nn.Linear(dict_size, input_dim, bias=False)

    def forward(self, x):
        # x: (batch, input_dim) — hidden layer activations
        # (batch, input_dim) -> (batch, dict_size)
        encoded = torch.relu(self.encoder(x))
        # (batch, dict_size) -> (batch, input_dim)
        decoded = self.decoder(encoded)
        return encoded, decoded

    def loss(self, x, encoded, decoded):
        # x: original activations, encoded: sparse features, decoded: reconstruction
        recon_loss = (decoded - x).pow(2).mean()
        l1_loss = encoded.abs().mean()
        return recon_loss + self.l1_coeff * l1_loss, recon_loss, l1_loss

    @torch.no_grad()
    def normalize_decoder(self):
        # constrain decoder columns to unit norm — prevents loss gaming
        # decoder.weight: (input_dim, dict_size)
        norms = self.decoder.weight.norm(dim=0, keepdim=True).clamp(min=1e-8)
        self.decoder.weight.div_(norms)

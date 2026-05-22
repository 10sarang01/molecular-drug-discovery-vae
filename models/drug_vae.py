import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool
class AdvancedDrugVAE(nn.Module):
    def __init__(self, in_features, hidden_dim, latent_dim):
        super(AdvancedDrugVAE, self).__init__()
        self.encoder_gat = GATv2Conv(in_features, hidden_dim, heads=4, concat=False, dropout=0.1)
        self.mu_layer = GATv2Conv(hidden_dim, latent_dim, heads=1, concat=False)
        self.logvar_layer = GATv2Conv(hidden_dim, latent_dim, heads=1, concat=False)
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1) 
        )

    def encode(self, x, edge_index):
        h = F.elu(self.encoder_gat(x, edge_index))
        mu = self.mu_layer(h, edge_index)
        logvar = self.logvar_layer(h, edge_index)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode_edges(self, z):
        return torch.sigmoid(torch.matmul(z, z.t()))

    def forward(self, x, edge_index, batch_index):
        mu, logvar = self.encode(x, edge_index)
        z = self.reparameterize(mu, logvar)
        
        reconstructed_adj = self.decode_edges(z)
        pooled_molecule = global_mean_pool(z, batch_index)
        property_prediction = self.classifier(pooled_molecule)
        return reconstructed_adj, property_prediction, mu, logvar

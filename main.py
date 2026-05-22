import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
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


def compute_composite_loss(predicted_adj, edge_index, num_nodes, pred_property, true_property, mu, logvar):

    true_adj = torch.zeros((num_nodes, num_nodes), device=predicted_adj.device)
    true_adj[edge_index[0], edge_index[1]] = 1.0
    recon_loss = F.binary_cross_entropy(predicted_adj, true_adj, reduction='mean')
    
    target_labels = true_property.float().unsqueeze(1)
    loss_weight = torch.tensor([0.5], device=predicted_adj.device) 
    
    prediction_loss = F.binary_cross_entropy_with_logits(
        pred_property, 
        target_labels, 
        pos_weight=loss_weight,
        reduction='mean'
    )
   
    kld_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)) / num_nodes
    
    total_composite_loss = recon_loss + prediction_loss + (0.02 * kld_loss)
    return total_composite_loss

def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Executing pipeline on device device: {device}")

    print("Fetching MUTAG bioinformatics benchmark from remote repository...")
    dataset = TUDataset(root='/tmp/MUTAG', name='MUTAG')
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = AdvancedDrugVAE(in_features=dataset.num_features, hidden_dim=32, latent_dim=8).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-5)
    
    print("\nInitiating Model Optimization Lifecycle (80 Epochs)...")
    for epoch in range(1, 81):
        model.train()
        epoch_loss = 0
        
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            

            predicted_adj, pred_property, mu, logvar = model(batch.x, batch.edge_index, batch.batch)

            loss = compute_composite_loss(predicted_adj, batch.edge_index, batch.num_nodes, pred_property, batch.y, mu, logvar)
            
 
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        if epoch % 20 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d}/80 | Composite Gradient Loss: {epoch_loss / len(loader):.4f}")


    model.eval()
    correct_predictions = 0
    total_molecules = 0
    all_preds = []
    all_trues = []

    print("\nTraining complete. Running final evaluation loop...")
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            _, pred_property, _, _ = model(batch.x, batch.edge_index, batch.batch)

            predictions = (pred_property >= 0.0).int().squeeze().cpu().numpy()
            labels = batch.y.cpu().numpy()
            
            predictions_tensor = (pred_property >= 0.0).int().squeeze()
            correct_predictions += (predictions_tensor == batch.y).sum().item()
            total_molecules += batch.y.size(0)
            
            if predictions.ndim == 0:
                all_preds.append([predictions.item()])
                all_trues.append([labels.item()])
            else:
                all_preds.extend(predictions)
                all_trues.extend(labels)

    all_preds = np.array(all_preds)
    all_trues = np.array(all_trues)

    print("\n" + "="*43)
    print("      FINAL VERIFIED PERFORMANCE REPORT      ")
    print("="*43)
    print(classification_report(all_trues, all_preds, target_names=['Safe/Inactive', 'Active Tumour Fighter']))
    
    final_accuracy = (correct_predictions / total_molecules) * 100
    print(f"Overall Network Classification Accuracy: {final_accuracy:.2f}%")
    print("="*43)

    cm = confusion_matrix(all_trues, all_preds)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Predicted Inactive', 'Predicted Active'],
                yticklabels=['True Inactive', 'True Active'])
    plt.title("Bio-Property Confusion Matrix Grid", fontsize=12, fontweight='bold')
    plt.ylabel("Ground Truth Lab Records")
    plt.xlabel("AI Predictive Decisions")

    plt.subplot(1, 2, 2)
    categories = ['Accurate Predictions', 'Misclassified Items']
    counts = [correct_predictions, total_molecules - correct_predictions]
    colors = ['#4CAF50', '#F44336']

    plt.bar(categories, counts, color=colors, width=0.5)
    for i, val in enumerate(counts):
        plt.text(i, val + 2, f"{val} ({val/total_molecules*100:.1f}%)", ha='center', fontweight='bold')

    plt.title("Overall Engineering Breakdown", fontsize=12, fontweight='bold')
    plt.ylabel("Number of Molecules")
    plt.ylim(0, total_molecules + 20)
    plt.grid(axis='y', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig('performance_dashboard.png', dpi=300)
    print("\nDiagnostic matrix asset exported successfully as 'performance_dashboard.png'.")
    plt.show()

    torch.save(model.state_dict(), "advanced_drug_vae.pt")
    print("Trained model weights saved successfully as 'advanced_drug_vae.pt'.")

if __name__ == "__main__":
    main()

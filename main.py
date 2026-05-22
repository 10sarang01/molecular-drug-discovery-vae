import torch
from data_loader import get_mutag_loaders
from models.drug_vae import AdvancedDrugVAE
from utils.loss import compute_composite_loss
from utils.visualization import generate_performance_dashboard

def main():

    loader, num_features = get_mutag_loaders(batch_size=32)

    model = AdvancedDrugVAE(in_features=num_features, hidden_dim=32, latent_dim=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-5)
    
    print("Initiating Multi-Task Model Optimization Lifecycle (80 Epochs)...")
    for epoch in range(1, 81):
        model.train()
        for batch in loader:
            optimizer.zero_grad()
            predicted_adj, pred_property, mu, logvar = model(batch.x, batch.edge_index, batch.batch)
            loss, _, _ = compute_composite_loss(predicted_adj, batch.edge_index, batch.num_nodes, pred_property, batch.y, mu, logvar)
            loss.backward()
            optimizer.step()
            
        if epoch % 20 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d}/80 Successfully Processed.")

    model.eval()
    correct_predictions, total_molecules = 0, 0
    with torch.no_grad():
        for batch in loader:
            _, pred_property, _, _ = model(batch.x, batch.edge_index, batch.batch)
            predictions = (pred_property >= 0.0).int().squeeze()
            correct_predictions += (predictions == batch.y).sum().item()
            total_molecules += batch.y.size(0)

    generate_performance_dashboard(model, loader, correct_predictions, total_molecules)
    
    torch.save(model.state_dict(), "advanced_drug_vae.pt")
    print("Execution complete. Metrics visualized and model states exported safely.")

if __name__ == "__main__":
    main()

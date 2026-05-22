import torch
import torch.nn.functional as F
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
    return total_composite_loss, recon_loss, prediction_loss

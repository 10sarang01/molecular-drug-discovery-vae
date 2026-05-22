from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

def get_mutag_loaders(batch_size=32):
    dataset = TUDataset(root='/tmp/MUTAG', name='MUTAG')
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader, dataset.num_features

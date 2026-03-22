import torch
from torch.utils.data import DataLoader

def accuracy(model, dataset, device):
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    return correct / total
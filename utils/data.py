from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

def load_data():
    #transform = transforms.Compose([transforms.ToTensor()])
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
    ])
    
    train = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    return train, test

def create_clients(dataset, num_clients=10):
    data_indices = np.arange(len(dataset))
    np.random.shuffle(data_indices)

    split = np.array_split(data_indices, num_clients)

    client_indices = {i: split[i] for i in range(num_clients)}
    return client_indices


def get_client_loader(dataset, indices, batch_size=32):
    subset = Subset(dataset, indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=True)
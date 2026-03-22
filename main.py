import torch
from utils.data import load_data, create_clients, get_client_loader
from client.client import Client
from models.cnn import CNN
from selection.random_sel import random_selection
from experiments.run_experiment import run_federated
from utils.metrics import accuracy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
train_dataset, test_dataset = load_data()

# Create client splits
client_indices = create_clients(train_dataset, num_clients=10)

# Create clients with DataLoader
clients = {}
for i in client_indices:
    loader = get_client_loader(train_dataset, client_indices[i], batch_size=32)
    clients[i] = Client(loader, device)

# Model
model = CNN().to(device)

# Train
#global_model = run_federated(model, clients, random_selection, rounds=5, k=5)
#global_model = run_federated(
#    model,
#    clients,
#    random_selection,
#    rounds=20,   # ⬅️ increase
#    k=5
#)
# Evaluate
#acc = accuracy(global_model, test_dataset, device)
#print("Final Accuracy:", acc)

from selection.greedy_divfl import greedy_divfl
from experiments.run_experiment import run_federated_divfl

global_model = run_federated_divfl(
    model,
    clients,
    greedy_divfl,
    rounds=20,
    k=5
)


# Evaluate
acc = accuracy(global_model, test_dataset, device)
print("Final Accuracy:", acc)
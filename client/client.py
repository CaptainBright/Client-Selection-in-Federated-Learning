import torch
import torch.nn as nn
import torch.optim as optim

class Client:
    def __init__(self, dataloader, device):
        self.loader = dataloader
        self.device = device

    def train(self, model, epochs=2):
        model.train()
        optimizer = optim.SGD(model.parameters(), lr=0.05)
        criterion = nn.CrossEntropyLoss()

        for _ in range(epochs):
            for x, y in self.loader:
                x = x.to(self.device)
                y = y.to(self.device)

                optimizer.zero_grad()
                output = model(x)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()

        return model.state_dict()

    def get_update(self, global_model, epochs=1):
        import copy
        model = copy.deepcopy(global_model)
        model.train()

        optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
        criterion = torch.nn.CrossEntropyLoss()

        for _ in range(epochs):
            for x, y in self.loader:
                x, y = x.to(self.device), y.to(self.device)

                optimizer.zero_grad()
                output = model(x)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()

        # Compute update = local - global
        update = {}
        for key in global_model.state_dict():
            update[key] = model.state_dict()[key] - global_model.state_dict()[key]

        return update
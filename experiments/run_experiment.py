
import copy
from server.fedavg import fedavg

def run_federated(model, clients, selection_fn, rounds, k):
    global_model = model

    for r in range(rounds):
        selected = selection_fn(list(clients.keys()), k)

        updates = []
        for c in selected:
            #local_model = copy.deepcopy(global_model)
            local_model = copy.deepcopy(global_model).to(next(global_model.parameters()).device)
            updates.append(clients[c].train(local_model))

        global_model = fedavg(global_model, updates)
        print(f"Round {r+1} completed")

    return global_model



def run_federated_divfl(model, clients, selection_fn, rounds, k):
    global_model = model

    for r in range(rounds):
        print(f"Round {r+1}")

        # 🔥 Step 1: get updates from ALL clients
        client_updates = {}
        for c in clients:
            client_updates[c] = clients[c].get_update(global_model)

        # 🔥 Step 2: select diverse clients
        selected = selection_fn(client_updates, k)

        # 🔥 Step 3: aggregate ONLY selected updates
        updates = [client_updates[c] for c in selected]

        global_model = fedavg(global_model, updates)

    return global_model

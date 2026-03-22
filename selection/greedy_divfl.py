import torch

def flatten(update):
    return torch.cat([v.flatten() for v in update.values()])


def distance(u1, u2):
    return torch.norm(u1 - u2)


def greedy_divfl(client_updates, k):
    """
    client_updates: dict {client_id: update_dict}
    """

    flat_updates = {c: flatten(client_updates[c]) for c in client_updates}

    clients = list(client_updates.keys())

    selected = []
    remaining = clients.copy()

    # Step 1: pick one randomly (or max norm)
    first = remaining[0]
    selected.append(first)
    remaining.remove(first)

    # Step 2: greedy selection
    while len(selected) < k:
        best_client = None
        best_score = -1

        for c in remaining:
            score = sum(distance(flat_updates[c], flat_updates[s]) for s in selected)

            if score > best_score:
                best_score = score
                best_client = c

        selected.append(best_client)
        remaining.remove(best_client)

    return selected
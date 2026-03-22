import torch

def fedavg(global_model, updates):
    new_state = {}

    for key in global_model.state_dict():
        avg_update = torch.mean(
            torch.stack([u[key].float() for u in updates]), dim=0
        )
        new_state[key] = global_model.state_dict()[key] + avg_update

    global_model.load_state_dict(new_state)
    return global_model
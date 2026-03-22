
import random
import copy

def shapley_values(clients, model, utility_fn, train_fn, iterations=10):
    values = {i: 0 for i in range(len(clients))}

    for _ in range(iterations):
        perm = list(range(len(clients)))
        random.shuffle(perm)

        temp_model = copy.deepcopy(model)
        prev_score = utility_fn(temp_model)

        for idx in perm:
            temp_model = train_fn(temp_model, clients[idx])
            new_score = utility_fn(temp_model)

            values[idx] += (new_score - prev_score)
            prev_score = new_score

    return values

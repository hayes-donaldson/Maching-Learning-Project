from torch import nn


def build_model(input_size, output_size):
    return nn.Sequential(
        nn.Linear(input_size, 32), nn.ReLU(),
        nn.Linear(32, 16), nn.ReLU(),
        nn.Linear(16, output_size),
    )

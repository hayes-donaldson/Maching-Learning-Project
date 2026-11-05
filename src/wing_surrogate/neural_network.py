from torch import nn


def build_model(input_size, output_size):

    model = nn.Sequential(
        nn.Linear(input_size, 16), nn.ReLU(),
        nn.Linear(16, 16), nn.ReLU(),
        nn.Linear(16, 8), nn.ReLU(),
        nn.Linear(8, output_size)
    )

    return model
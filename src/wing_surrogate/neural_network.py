from torch import nn


def build_model(input_size=2, output_size=2):

    model = nn.Sequential(
        nn.Linear(input_size, 8), nn.ReLU(),
        nn.Linear(8, output_size)
    )

    return model
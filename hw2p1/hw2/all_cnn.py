import torch.nn as nn
from torch.nn import Sequential, Dropout, Conv2d, ReLU, AvgPool2d


class Flatten(nn.Module):
    """
    Implement a simple custom module that reshapes (n, m, 1, 1) tensors to (n, m).
    """

    def forward(self, x):
        """
        Implement this function
        :param x: input variable (n, m, 1, 1)
        :return: flattened variable (n, m)
        """
        return x.view(x.size()[0], -1)


def all_cnn_module():
    """
    Create a nn.Sequential model containing all of the layers as specified in the paper.
    Use a AvgPool2d to pool and then your Flatten layer to flatten.
    You should have a total of exactly 23 layers of types:
    - nn.Dropout
    - nn.Conv2d
    - nn.ReLU
    - nn.AvgPool2d
    - Flatten
    :return: a nn.Sequential model
    """
    return Sequential(
        Dropout(0.2),
        Conv2d(3, 96, (3, 3), padding=1),
        ReLU(),
        Conv2d(96, 96, (3, 3), padding=1),
        ReLU(),
        Conv2d(96, 96, (3, 3), padding=(1, 1), stride=(2, 2)),
        ReLU(),
        Dropout(0.5),
        Conv2d(96, 192, (3, 3), padding=(1, 1)),
        ReLU(),
        Conv2d(192, 192, (3, 3), padding=(1, 1)),
        ReLU(),
        Conv2d(192, 192, (3, 3), padding=(1, 1), stride=2),
        ReLU(),
        Dropout(0.5),
        Conv2d(192, 192, (3, 3)),
        ReLU(),
        Conv2d(192, 192, (1, 1)),
        ReLU(),
        Conv2d(192, 10, (1, 1)),
        ReLU(),
        AvgPool2d(6),
        Flatten()
    )

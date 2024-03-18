
import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):
    def __init__(self, input_size, hidden_layer, output_size):
        super(Network, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_layer)
        self.l2 = nn.Linear(hidden_layer, output_size)
    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.softmax(self.l2(x))
        return x
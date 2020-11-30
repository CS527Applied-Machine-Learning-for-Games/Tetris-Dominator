import torch.nn as nn


class DeepQNetwork(nn.Module):
    def __init__(self):

        super(DeepQNetwork, self).__init__()

        self.layer_1 = nn.Sequential(nn.Linear(5, 4), nn.ReLU(inplace=True))
        self.layer_2 = nn.Sequential(nn.Linear(4, 1))

        for m in self.modules():
            if type(m) == nn.Linear:
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        x = self.layer_1(x)
        x = self.layer_2(x)

        return x

import torch
from torch import nn


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        self.emb_dim = 300
        self.dis_layers = 2 #2 Discriminator layers
        self.dis_hid_dim = 2048 #Discriminator hidden layer dimensions
        self.dis_dropout = 0 #Discriminator dropout
        self.dis_input_dropout = 0.1 #Discriminator input dropout

        layers = [nn.Dropout(self.dis_input_dropout)]
        # network structure
        for i in range(self.dis_layers + 1):
            input_dim = self.emb_dim if i == 0 else self.dis_hid_dim
            output_dim = 1 if i == self.dis_layers else self.dis_hid_dim
            layers.append(nn.Linear(input_dim, output_dim))
            if i < self.dis_layers:
                layers.append(nn.LeakyReLU(0.2))
                layers.append(nn.Dropout(self.dis_dropout))
        layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        assert x.dim() == 2 and x.size(1) == self.emb_dim
        return self.layers(x).view(-1)
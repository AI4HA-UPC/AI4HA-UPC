import torch.nn as nn


class MLPRegressor(nn.Module):
    def __init__(self, input_size,  hidden_layers_sizes, activation=nn.ReLU):
        super(MLPRegressor, self).__init__()
        self.layers = []
        self.fc1 = nn.Linear(input_size, hidden_layers_sizes[0])
        self.layers.append(self.fc1)
        self.layers.append(activation())
        for i in range(1, len(hidden_layers_sizes)):
            self.layers.append(nn.Linear(hidden_layers_sizes[i-1], hidden_layers_sizes[i]))
            self.layers.append(activation())
        self.layers = nn.Sequential(*self.layers)       
        self.output = nn.Linear(hidden_layers_sizes[-1], 1)

    def forward(self, x):
        return self.output(self.layers(x.view(x.size(0), -1)))
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):

    def __init__(self, embed_n, outputs):
        super(DQN, self).__init__()
        self.gru = nn.GRU(input_size =embed_n, hidden_size= 16, num_layers=1)
        self.fc1 = nn.Linear(16,16)
        self.fc2 = nn.Linear(16,outputs)


    def forward(self, x):
        x , _ = self.gru(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x



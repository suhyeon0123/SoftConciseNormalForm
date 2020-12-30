import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQN(nn.Module):

    def __init__(self, num_symbols=12, embedding_dim=4, hidden_dim=64, num_actions=6):
        super(DQN, self).__init__()

        self.hidden_dim = hidden_dim

        self.symb_embeddings = nn.Embedding(num_symbols, embedding_dim)


        self.pos_rnn = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=1)
        self.neg_rnn = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=1)
        self.regex_rnn = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=1)

        self.fc1 = nn.Linear(self.hidden_dim * 3, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, num_actions)

    def initialize_hidden_state(self):
        return torch.zeros((1, self.batch_size, self.hidden_dim)).to(device)

    def forward(self, regex, pos, neg):

        self.batch_size = pos.shape[0]


        regex = regex.to(device)
        pos = pos.to(device)
        neg = neg.to(device)


        regex_embedded = self.symb_embeddings(regex)
        pos_embedded = self.symb_embeddings(pos)
        neg_embedded = self.symb_embeddings(neg)

        regex_embedded = regex_embedded.permute(1, 0, 2)
        pos_embedded = pos_embedded.permute(1, 0, 2)
        neg_embedded = neg_embedded.permute(1, 0, 2)


        regex_x, hidden_regex_x = self.regex_rnn(regex_embedded)
        pos_x , hidden_pos_x = self.pos_rnn(pos_embedded)
        neg_x, hidden_neg_x = self.neg_rnn(neg_embedded)

        concat_feature = torch.cat((pos_x[-1], neg_x[-1], regex_x[-1]), dim=-1)

        x = F.relu(self.fc1(concat_feature))
        x = self.fc2(x)
        return x



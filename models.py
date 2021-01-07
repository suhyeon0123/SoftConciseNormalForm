import torch
import torch.nn as nn
import torch.nn.functional as F

LENGTH_LIMIT = 30
EXAMPLE_LENGTH_LIMIT = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQN(nn.Module):

    def __init__(self, num_symbols=12, embedding_dim=4, hidden_dim=128, num_actions=12):
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
        pos_x, hidden_pos_x = self.pos_rnn(pos_embedded)
        neg_x, hidden_neg_x = self.neg_rnn(neg_embedded)

        concat_feature = torch.cat((pos_x[-1], neg_x[-1], regex_x[-1]), dim=-1)

        x = F.relu(self.fc1(concat_feature))
        x = self.fc2(x)
        return x

class DuelingDQN(nn.Module):

    def __init__(self, num_symbols=12, embedding_dim=4, hidden_dim=128, num_actions=12):
        super(DuelingDQN, self).__init__()

        self.hidden_dim = hidden_dim

        self.symb_embeddings = nn.Embedding(num_symbols, embedding_dim)

        self.pos_rnn = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=1)
        self.neg_rnn = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=1)
        self.regex_rnn = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=1)

        self.fc1 = nn.Linear(self.hidden_dim * 3, self.hidden_dim)

        self.advantage = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, num_actions)
        )

        self.value = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)
        )

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
        pos_x, hidden_pos_x = self.pos_rnn(pos_embedded)
        neg_x, hidden_neg_x = self.neg_rnn(neg_embedded)

        concat_feature = torch.cat((pos_x[-1], neg_x[-1], regex_x[-1]), dim=-1)

        x = F.relu(self.fc1(concat_feature))

        advantage = self.advantage(x)
        value = self.value(x)

        return value + advantage  - advantage.mean()


class ACNet(nn.Module):

    def __init__(self, num_symbols=12, embedding_dim=4, hidden_dim=128, num_actions=6):
        super(ACNet, self).__init__()

        self.hidden_dim = hidden_dim

        self.symb_embeddings = nn.Embedding(num_symbols, embedding_dim)

        self.pos_rnn = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=1)
        self.neg_rnn = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=1)
        self.regex_rnn = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=1)

        self.fc1 = nn.Linear(self.hidden_dim * 3, self.hidden_dim)

        self.advantage = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, num_actions)
        )

        self.value = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)
        )

    def initialize_hidden_state(self):
        return torch.zeros((1, self.batch_size, self.hidden_dim)).to(device)

    def act(self, x):
        '''상태 x로부터 행동을 확률적으로 결정'''
        value, actor_output = self(x[:,:LENGTH_LIMIT], x[:,LENGTH_LIMIT:LENGTH_LIMIT + EXAMPLE_LENGTH_LIMIT], x[:,LENGTH_LIMIT + EXAMPLE_LENGTH_LIMIT:])
        # actor_output=torch.clamp_min_(actor_output,min=0)
        # dim=1이므로 행동의 종류에 대해 softmax를 적용
        action_probs = F.softmax(actor_output, dim=1)
        action = action_probs.multinomial(num_samples=1)  # dim=1이므로 행동의 종류에 대해 확률을 계산
        return action

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
        pos_x, hidden_pos_x = self.pos_rnn(pos_embedded)
        neg_x, hidden_neg_x = self.neg_rnn(neg_embedded)

        concat_feature = torch.cat((pos_x[-1], neg_x[-1], regex_x[-1]), dim=-1)

        x = F.relu(self.fc1(concat_feature))

        advantage = self.advantage(x)
        value = self.value(x)

        return value, advantage

    def get_value(self, x):
        '''상태 x로부터 상태가치를 계산'''
        value, actor_output = self(x[:,:LENGTH_LIMIT], x[:,LENGTH_LIMIT:LENGTH_LIMIT + EXAMPLE_LENGTH_LIMIT], x[:,LENGTH_LIMIT + EXAMPLE_LENGTH_LIMIT:])

        return value


    def evaluate_actions(self, x, actions):
        '''상태 x로부터 상태가치, 실제 행동 actions의 로그 확률, 엔트로피를 계산'''
        value, actor_output = self(x[:,:LENGTH_LIMIT], x[:,LENGTH_LIMIT:LENGTH_LIMIT + EXAMPLE_LENGTH_LIMIT], x[:,LENGTH_LIMIT + EXAMPLE_LENGTH_LIMIT:])

        log_probs = F.log_softmax(actor_output, dim=1)  # dim=1이므로 행동의 종류에 대해 확률을 계산
        action_log_probs = log_probs.gather(1, actions.detach())  # 실제 행동의 로그 확률(log_probs)을 구함

        probs = F.softmax(actor_output, dim=1)  # dim=1이므로 행동의 종류에 대한 계산
        entropy = -(log_probs * probs).sum(-1).mean()

        return value, action_log_probs, entropy

    def mish(self, x):
        return x * torch.tanh(F.softplus(x))
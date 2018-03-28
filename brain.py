import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.autograd import Variable
import random


class NeuralNet(nn.Module):
    """
    The underlying neural network.
    """

    def __init__(self, num_features, num_actions):
        super().__init__()
        self.c1 = nn.Linear(num_features, 30)
        self.c2 = nn.Linear(30, 30)
        self.c3 = nn.Linear(30, num_actions)

    def forward(self, state):
        return self.c3(F.relu(self.c2(F.relu(self.c1(state)))))


class ExpReplay(object):
    """
    Implementation of Experience Replay
    """

    def __init__(self, cap):
        self.cap = cap
        self.mem = []

    def push(self, event):
        self.mem.append(event)
        if len(self.mem) > self.cap:
            del self.mem[0]

    def sample(self, batch_size):
        samples = zip(*random.sample(self.mem, batch_size))
        return [Variable(torch.cat(x, 0)) for x in samples]

    def __len__(self):
        return len(self.mem)



class DeepQNet(object):
    def __init__(self, num_features, num_actions, gamma):
        self.net = NeuralNet(num_features, num_actions)
        self.gamma = gamma
        self.exp = ExpReplay(100000)
        self.opt = optim.Adam(self.net.parameters(), lr=0.001)
        self.last_state = torch.Tensor(num_features).unsqueeze(0)
        self.last_action = 0
        self.last_reward = 0

    def update(self, reward, signal):
        current_state = torch.Tensor(signal).float().unsqueeze(0)
        self.exp.push(
            (self.last_state, current_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))
        action = self.select_action(current_state)
        self.learn()
        self.last_action = action
        self.last_state = current_state
        self.last_reward = reward
        return action

    def select_action(self, state):
        out = self.net(Variable(state, volatile=True))
        probs = F.softmax(out*10)
        return probs.multinomial().data[0, 0]

    def learn(self):
        if len(self.exp) < 10:
            return
        batch_state, batch_next_state, batch_action, batch_reward = self.exp.sample(10)
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        target = self.gamma * next_outputs + batch_reward
        td_loss = F.smooth_l1_loss(outputs, target)
        self.opt.zero_grad()
        td_loss.backward(retain_variables=True)
        self.opt.step()

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.autograd import Variable
import random
from datetime import datetime
import os


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
    def __init__(self, num_features, num_actions, gamma, path=None):
        self.net = NeuralNet(num_features, num_actions)
        self.opt = optim.Adam(self.net.parameters(), lr=0.001)

        # Load checkpoint from disk
        if path and os.path.isfile(path):
            print(self)
            try:
                ckpt = torch.load(path)
                self.net.load_state_dict(ckpt['net_state'])
                self.opt.load_state_dict(ckpt['opt_state'])
                print("Checkpoint at {} loaded...".format(path))
            except Exception as e:
                print(e)
                print("Cannot load checkpoint...")

        self.gamma = gamma
        self.exp = ExpReplay(100000)
        self.last_state = {
            2: torch.Tensor(num_features).unsqueeze(0),
            3: torch.Tensor(num_features).unsqueeze(0)
        }
        self.last_action = {2: 0, 3: 0}
        self.last_reward = {2: 0, 3: 0}

    def update(self, reward, current_signal):
        team = current_signal[0]
        current_state = torch.Tensor(current_signal).float().unsqueeze(0)
        self.exp.push(
            (self.last_state[team], current_state,
            torch.LongTensor([int(self.last_action[team])]), torch.Tensor([self.last_reward[team]])))
        action = self.select_action(current_state)
        self.learn(100)
        self.last_action[team] = action
        self.last_state[team] = current_state
        self.last_reward[team] = reward
        return action

    def select_action(self, state):
        out = self.net(Variable(state, volatile=True))
        probs = F.softmax(out*100)
        return probs.multinomial().data[0, 0]

    def learn(self, batch_size):
        if len(self.exp) < batch_size:
            return
        batch_state, batch_next_state, batch_action, batch_reward = self.exp.sample(batch_size)
        outputs = self.net(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        next_outputs = self.net(batch_next_state).detach().max(1)[0]
        target = self.gamma * next_outputs + batch_reward
        td_loss = F.smooth_l1_loss(outputs, target)
        self.opt.zero_grad()
        td_loss.backward(retain_variables=True)
        self.opt.step()


    def save(self):
        torch.save({"net_state": self.net.state_dict(),
                    "opt_state" : self.opt.state_dict(),
                   }, datetime.now().strftime("ckpt %Y-%m-%d %H.%M.%S.brain"))
        print("Checkpoint saved.")

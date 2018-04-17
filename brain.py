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
        """
        Initialize the network with:
            - Input layer with `num_features` nodes
            - Hidden layer 1 with 100 nodes
            - Hidden layer 2 with 100 nodes
            - Output layer with `num_actions` nodes

        """
        super().__init__()
        self.c1 = nn.Linear(num_features, 100)
        self.c2 = nn.Linear(100, 100)
        self.c3 = nn.Linear(100, num_actions)

    def forward(self, state):
        """Uses rectified linear unit to feed data through the network."""
        return self.c3(F.relu(self.c2(F.relu(self.c1(state)))))


class ExpReplay(object):
    """
    Implementation of Experience Replay. This is a list of transitions with a capacity.
    In Atari paper, it was 100000.

    Each transition is in form of 4-element tuple:
        (Past State, Future State, Action Performed, Reward received going from Past state to Future state)
    """

    def __init__(self, cap):
        self.cap = cap
        self.mem = []

    def push(self, event):
        """Push the event to list, remove the oldest one if capacity is reached."""
        self.mem.append(event)
        if len(self.mem) > self.cap:
            del self.mem[0]

    def sample(self, batch_size):
        """
        Sample `batch_size` number of transitions, change up the Tensor dimension
        to match the neural network requirement.
        """
        samples = zip(*random.sample(self.mem, batch_size))
        return [Variable(torch.cat(x, 0)) for x in samples]

    def __len__(self):
        return len(self.mem)


class DeepQNet(object):
    """
    Deep Q Network implementation
    """

    def __init__(self, num_features, num_actions, gamma, path=None):

        # Initialize NeuralNet with number of input and output nodes
        self.net = NeuralNet(num_features, num_actions)

        # Optimizer that help with backpropagation
        self.opt = optim.Adam(self.net.parameters())

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
        # Range 0-1
        self.gamma = gamma

        # Experience replay pool of 100000
        self.exp = ExpReplay(100000)

        # Set some initial states
        self.last_state = {
            2: torch.Tensor(num_features).unsqueeze(0),
            3: torch.Tensor(num_features).unsqueeze(0)
        }
        self.last_action = {2: 0, 3: 0}
        self.last_reward = {2: 0, 3: 0}

    def update(self, team, reward, current_signal):
        current_state = torch.FloatTensor(current_signal).unsqueeze(0)
        self.exp.push(
            (self.last_state[team], current_state,
            torch.LongTensor([int(self.last_action[team])]), torch.Tensor([self.last_reward[team]])))
        action = self.select_action(current_state)
        self.learn(50)
        self.last_action[team] = action
        self.last_state[team] = current_state
        self.last_reward[team] = reward
        return action

    def select_action(self, state):
        """Selects actions based on Q-values."""

        # Gets Q-values of all action
        q_values = self.net(Variable(state, volatile=True))

        # Pass the result through softmax to get distribution that sums
        # up to 1. The number is tempurature, it there to give the softmax
        # function more confidence
        probabilities = F.softmax(q_values)

        # Use multinomial function to sample from the distribution
        return probabilities.multinomial().data[0, 0]

    def learn(self, batch_size):
        """
        Extract `batch_size` number of transitions from Experience Replay pool to
        train the neral network.
        """

        # If the pool does not have enough transitions, do no training
        if len(self.exp) < batch_size:
            return

        # Extract the transitions from Experience Replay pool
        batch_state, batch_next_state, batch_action, batch_reward = self.exp.sample(batch_size)

        # Gets the Q-values of the performed actions
        outputs = self.net(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)

        # Gets the max Q-values of future states
        next_outputs = self.net(batch_next_state).detach().max(1)[0]

        # Calculate the target Q-values
        target = self.gamma * next_outputs + batch_reward

        # Calculate temporal difference between the old outputs vs. the newly found target Q-values
        td_error = F.smooth_l1_loss(outputs, target)

        # Back propagations using optimizer happen here
        self.opt.zero_grad()
        td_error.backward(retain_variables=True)
        self.opt.step()


    def save(self):
        """Save a checkpoint."""
        torch.save({"net_state": self.net.state_dict(),
                    "opt_state" : self.opt.state_dict(),
                   }, datetime.now().strftime("ckpt %Y-%m-%d %H.%M.%S.brain"))
        print("Checkpoint saved.")

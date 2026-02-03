"""
Double DQN Agent for Lunar Lander
Implements Double Deep Q-Network to reduce overestimation bias
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque, namedtuple


class QNetwork(nn.Module):
    """Neural network for Q-value approximation."""

    def __init__(self, state_size, action_size, seed, hidden_size=256):
        """
        Initialize parameters and build model.

        Args:
            state_size (int): Dimension of state space
            action_size (int): Dimension of action space
            seed (int): Random seed
            hidden_size (int): Size of hidden layers
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        # Network architecture: state -> hidden1 -> hidden2 -> hidden3 -> actions
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc4 = nn.Linear(hidden_size // 2, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object."""
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience",
                                    field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float()
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long()
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float()
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float()
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float()

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class DoubleDQNAgent:
    """Double DQN Agent - reduces overestimation bias."""

    def __init__(self, state_size, action_size, seed=0,
                 buffer_size=int(1e5), batch_size=64, gamma=0.99,
                 tau=1e-3, lr=5e-4, update_every=4, hidden_size=256):
        """
        Initialize a DoubleDQNAgent.

        Args:
            state_size (int): Dimension of state space
            action_size (int): Dimension of action space
            seed (int): Random seed
            buffer_size (int): Replay buffer size
            batch_size (int): Minibatch size
            gamma (float): Discount factor
            tau (float): Soft update parameter
            lr (float): Learning rate
            update_every (int): How often to update the network
            hidden_size (int): Size of hidden layers
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.update_every = update_every

        # Detect device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Q-Networks (local and target)
        self.qnetwork_local = QNetwork(state_size, action_size, seed, hidden_size).to(self.device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed, hidden_size).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)

        # Replay memory
        self.memory = ReplayBuffer(action_size, buffer_size, batch_size, seed)

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        """Save experience and learn if enough samples available."""
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)

    def act(self, state, eps=0.):
        """
        Returns actions for given state as per current policy.

        Args:
            state (array_like): Current state
            eps (float): Epsilon for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """
        Update value parameters using given batch of experience tuples.
        Uses Double DQN to reduce overestimation bias.
        """
        states, actions, rewards, next_states, dones = experiences
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Double DQN: Use local network to select actions, target network to evaluate
        # Select best actions for next states using LOCAL network
        next_actions = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1)

        # Evaluate those actions using TARGET network
        Q_targets_next = self.qnetwork_target(next_states).gather(1, next_actions)

        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients to prevent catastrophic weight updates
        torch.nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Update target network
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

    def soft_update(self, local_model, target_model, tau):
        """
        Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def save(self, filename):
        """Save model weights."""
        torch.save(self.qnetwork_local.state_dict(), filename)

    def load(self, filename):
        """Load model weights."""
        self.qnetwork_local.load_state_dict(torch.load(filename, map_location=self.device))
        self.qnetwork_target.load_state_dict(torch.load(filename, map_location=self.device))

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torch.utils.tensorboard as tb
import os


class DeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, name, input_dims, chkpt_dir):
        super(DeepQNetwork, self).__init__()
        self.checkpoint_file = os.path.join(chkpt_dir, name)
        self.cnv1 = nn.Conv2d(input_dims[0], 32, 8, stride=4, padding=0)
        self.cnv2 = nn.Conv2d(32, 64, 4, stride=2, padding=0)
        self.cnv3 = nn.Conv2d(64, 128, 3, stride=1, padding=0)
        self.fc1 = nn.Linear(128 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        #state = T.tensor(state, dtype=T.float).to(self.device)
        x = F.relu(self.cnv1(state))
        x = F.relu(self.cnv2(x))
        x = F.relu(self.cnv3(x))
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        actions = self.fc2(x)

        return actions


class Agent():
    def __init__(self, current_game, gamma, epsilon, lr, n_actions, input_dims, mem_size, batch_size, eps_min=0.01, eps_dec=5e-7,
                 chkpt_dir='tmp/'):
        self.current_game = current_game.lstrip('ALE/')
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.chkpt_dir = chkpt_dir
        self.action_space = [i for i in range(n_actions)]
        self.learn_step_counter = 0
        self.states = np.zeros((mem_size, *input_dims), dtype=np.float32)
        self.actions = np.zeros(mem_size, dtype=np.int32)
        self.rewards = np.zeros(mem_size, dtype=np.float32)
        self.states_ = np.zeros((mem_size, *input_dims), dtype=np.float32)
        self.dones = np.zeros(mem_size, dtype=np.bool_)
        self.mem_ctr = 0
        self.episode_cntr = 0
        self.iter_cntr = 0
        self.mem_size = mem_size
        self.summary_writer = tb.SummaryWriter()

        self.policy = DeepQNetwork(lr, n_actions, input_dims=input_dims, name=self.current_game + 'policy', chkpt_dir=chkpt_dir)
        self.target = DeepQNetwork(lr, n_actions, input_dims=input_dims, name=self.current_game + 'target', chkpt_dir=chkpt_dir)

        self.startup()

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_ctr % self.mem_size
        self.states[index] = state
        self.actions[index] = action
        self.rewards[index] = reward
        self.states_[index] = state_
        self.dones[index] = done
        self.mem_ctr += 1

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor(np.array([observation]), dtype=T.float).to(self.policy.device)
            actions = self.policy.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def play(self, observation):
        state = T.tensor(np.array([observation]), dtype=T.float, device=self.policy.device)
        actions = self.policy.forward(state)
        action = T.argmax(actions).item()
        return action

    def learn(self):
        if self.mem_ctr < self.batch_size:
            return

        self.policy.optimizer.zero_grad()

        num_mems = min(self.mem_ctr, self.mem_size)

        batch = np.random.choice(num_mems, self.batch_size, replace=False)
        batch_index = np.arange(self.batch_size)

        states = T.tensor(self.states[batch], device=self.policy.device)
        new_states = T.tensor(self.states_[batch], device=self.policy.device)
        actions = T.tensor(self.actions[batch], device=self.policy.device)
        rewards = T.tensor(self.rewards[batch], device=self.policy.device)
        dones = T.tensor(self.dones[batch], device=self.policy.device)

        current_qs = self.policy.forward(states)[batch_index, actions]
        next_qs = self.target.forward(new_states)
        next_qs[dones] = 0.0

        target_q = rewards + self.gamma * T.max(next_qs, dim=1)[0]

        loss = self.policy.loss(current_qs, target_q).to(self.policy.device)
        self.summary_writer.add_scalar('Loss', loss, self.iter_cntr)
        self.summary_writer.flush()
        loss.backward()
        self.policy.optimizer.step()
        self.iter_cntr += 1
        if self.iter_cntr % 40000 == 0:
            self.update_target_network()

        if self.epsilon > self.eps_min:
            self.epsilon -= self.eps_dec

    def update_target_network(self):
        self.target.load_state_dict(self.policy.state_dict())

    def close_writer(self):
        self.summary_writer.close()

    def save_checkpoint(self):
        filepath = self.policy.checkpoint_file
        target_filepath = self.target.checkpoint_file
        print('... saving checkpoint ...')
        T.save({
            'state_dict': self.policy.state_dict(),
            'optimizer': self.policy.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'episode': self.episode_cntr,
            'steps': self.iter_cntr
        }, filepath)
        T.save(self.target.state_dict(), target_filepath)

    def load_checkpoint(self):
        filepath = self.policy.checkpoint_file
        target_filepath = self.target.checkpoint_file
        checkpoint = T.load(filepath)
        self.policy.load_state_dict(checkpoint['state_dict'])
        self.policy.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.episode_cntr = checkpoint['episode']
        self.target.load_state_dict(self.policy.state_dict())
        self.iter_cntr = checkpoint['steps']
        self.target.load_state_dict(T.load(target_filepath))
        print('Loaded checkpoint from', filepath)
        print('Epsilon:', self.epsilon)
        print('Episodes:', self.episode_cntr)

    def startup(self):
        if not os.path.exists(self.chkpt_dir):
            os.mkdir(self.chkpt_dir)
        if os.path.exists(self.policy.checkpoint_file) and os.path.exists(self.target.checkpoint_file):
            print('Checkpoint found...')
            self.load_checkpoint()
        else:
            print('No checkpoint found...')

    def increment_episode(self):
        self.episode_cntr += 1

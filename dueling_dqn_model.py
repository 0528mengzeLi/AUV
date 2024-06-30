# models/dueling_dqn_agent.py
import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

class DuelingDQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DuelingDQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 24)

        self.value_fc = nn.Linear(24, 24)
        self.value = nn.Linear(24, 1)

        self.advantage_fc = nn.Linear(24, 24)
        self.advantage = nn.Linear(24, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))

        value = torch.relu(self.value_fc(x))
        value = self.value(value)

        advantage = torch.relu(self.advantage_fc(x))
        advantage = self.advantage(advantage)

        advantage_mean = torch.mean(advantage, dim=1, keepdim=True)
        advantage = advantage - advantage_mean
        q_values = value + advantage

        return q_values

class DuelingDQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # 折扣因子
        self.epsilon = 1.0  # 探索率
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DuelingDQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def remember(self, state, action, reward, next_state, done):
        # 记忆经验
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # 根据当前状态选择动作
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            act_values = self.model(state)
        return torch.argmax(act_values[0]).item()

    def replay(self, batch_size):
        # 经验回放，从记忆中采样并训练模型
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state).to(self.device)
            next_state = torch.FloatTensor(next_state).to(self.device)
            action = torch.tensor([[action]]).to(self.device)
            reward = torch.tensor([[reward]]).to(self.device)
            done = torch.tensor([[done]]).to(self.device)

            target = reward
            if not done:
                target = reward + self.gamma * torch.max(self.model(next_state)).unsqueeze(0)
            target_f = self.model(state)
            target_f[0][action] = target

            self.model.train()
            output = self.model(state)
            loss = nn.MSELoss()(output, target_f)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        # 加载模型权重
        self.model.load_state_dict(torch.load(name))

    def save(self, name):
        # 保存模型权重
        torch.save(self.model.state_dict(), name)

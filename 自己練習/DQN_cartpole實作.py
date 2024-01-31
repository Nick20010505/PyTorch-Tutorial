import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym

# hyperparameters
BATCH_SIZE = 32
LR = 0.01
EPSILON = 0.9
GAMMA = 0.9
TARGET_REPLACE_ITER = 100    # 進行 hard update 的步數
MEMORY_CAPACITY = 2000       # replay buffer 的大小
env = gym.make('CartPole-v0')
env = env.unwrapped
N_ACTIONS = env.action_space.n
N_STATES = env.observation_space.shape[0]
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape     # to confirm the shape

# 定義網路形式
class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 50)
        self.fc1.weight.data.normal_(0, 0.1)   # 初始化網路參數
        self.out = nn.Linear(50, N_ACTIONS)    # 輸出為動作的維度
        self.out.weight.data.normal_(0, 0.1)   # 初始化網路參數

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value
    
# 定義 DQN 的內容 (整個流程)
class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()

        self.learn_step_counter = 0    # 計算 target 是否需要被更新了
        self.memory_counter = 0        # 計算目前 replay buffer 儲存到哪個位置
        self.memory = np.zeros((MEMORY_CAPACITY,N_STATES*2 + 2))  # 創建一個 replay buffer，寬度為(s, a, r, s_)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        if np.random.uniform() < EPSILON:    # greedy policy
            actions_value = self.eval_net.forward(x)  
            action = torch.max(actions_value, 1)[1].data.numpy() # 選Q值最大的action
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)  # return the argmax index
        else:
            action = np.random.randint(0, N_ACTIONS)
            action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)  # return the argmax index
        return action
    
    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))  # 將經驗存成一個 transition
        index = self.memory_counter % MEMORY_CAPACITY  # 計算該經驗目前應該放到哪裡
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target network 更新 --> hard update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # 抓出一個 batch 的經驗
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]    # random sample batch
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])  # 所有列的前 n 行
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES + 1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])

        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()     # 防止更新網路，detach 是直接抓下來
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)  # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

dqn = DQN()

print('\nCollecting experience...')
for i_episode in range(400):
    # 沒有 info 的話，會多一維(array([ 0.01663968, -0.0364228 , -0.03484314, -0.01153948], dtype=float32), {})
    s, info = env.reset()     # s = env.reset() 的話 要加一行 s = np.array(s[0])
    ep_r = 0
    while True:
        env.render()
        a = dqn.choose_action(s)  # 選擇動作

        s_, r, done, info, _ = env.step(a)  # 執行動作，這邊改為 5 個 Output

        # modify the reward
        x, x_dot, theta, theta_dot = s_
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        r = r1 + r2

        dqn.store_transition(s, a, r, s_)

        ep_r += r
        if dqn.memory_counter > MEMORY_CAPACITY:
            dqn.learn()
            if done:
                print('Ep: ', i_episode,
                      '| Ep_r: ', round(ep_r, 2))
        if done:
            break
        s = s_
from model import Model

import torch as tc
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import pickle
from tqdm import tqdm
import copy


DEVICE = tc.device("cuda:1" if tc.cuda.is_available() else "cpu")

class Qnn(nn.Module):
    def __init__(self, input_layer):
        super(Qnn, self).__init__()
        self.net = nn.Sequential(nn.Linear(input_layer, input_layer), 
            # nn.ReLU(), nn.Linear(2*input_layer, 2*input_layer),
            nn.ReLU(), nn.Linear(input_layer, input_layer),
            nn.ReLU(), nn.Linear(input_layer, 1)).to(DEVICE)
        tc.manual_seed(0)
        self.net.apply(self.weights_init)

        self.input_layer = input_layer

    def forward(self, x):
        # y = np.append(x,a)
        x = tc.tensor(x, dtype=tc.float, device=DEVICE)
        return self.net(x)
        
    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            tc.nn.init.normal_(m.weight, mean=0.0, std=0.5)
            # tc.nn.init.zero_(m.bias)



def dqn(model, user_class, Q, Q_hat, learning_rate=0.01, num_episodes=1000, batch_size=10, discount_factor=1., learnign_rate_decay=0.9, e_greedy=0.01):
    total_rewards, batch_rewards, batch_states_fast, batch_actions = [], [], [], []
    batch_counter = 1

    optimizer = tc.optim.SGD(Q.parameters(), lr=learning_rate)
    scheduler = tc.optim.lr_scheduler.ExponentialLR(optimizer, gamma=learnign_rate_decay)
    
    history = []

    for episode in tqdm(range(num_episodes)):
        state_slow, state_fast = model.reset(user_class)
        rewards, actions, states_slow, states_fast = [], [], [], []

        while True:
            # use policy to make predictions and run an action
            # state_fast = tc.FloatTensor(state_fast).to(DEVICE)
            

            if np.random.uniform() > e_greedy:
                # print(np.tile(state_fast, (model.A_d,1)).shape)
                x = np.hstack([np.tile(state_fast, (model.A_d,1)), np.arange(model.A_d).reshape(model.A_d,1)])
                Q_list = Q_hat(x).detach().cpu().numpy()
                action = np.argmax(Q_list)
            else:
                action = np.random.choice(model.A_d, 1)
                
            # action_probs = policy(state_fast).detach().cpu().numpy()
            # # action = [np.argmax(action_probs)]
            # action = [ np.random.choice(len(action_probs), 1, p=action_probs) ]
            action_vec = np.zeros(model.A_d)
            action_vec[action] = 1

            pre_state_fast = state_fast
            state_slow, state_fast, terminal = model.update(state_slow, state_fast, action_vec, user_class)
            reward = model.reward(state_slow, state_fast, action_vec)
            
            rewards.append(reward)
            history.append( (pre_state_fast, action, reward, state_fast) )

            batch_counter += 1
            if batch_counter % batch_size >= 0:
                # sample from history
                q_index = np.random.choice(len(history), batch_size)
                s_ps, s_a, s_r, s_s = history[q_index[0]]
                x = np.hstack([np.tile(s_s, (model.A_d,1)), np.arange(model.A_d).reshape(model.A_d,1)])
                Q_list = Q_hat(x).detach().cpu().numpy()
                y = s_r if terminal else s_r+discount_factor*max(Q_list)
                y = tc.tensor(y, dtype=tc.float, device=DEVICE)

                optimizer.zero_grad()
                loss = (y - Q(np.append(s_ps, action)))**2
                loss.backward()
                optimizer.step()
                Q_hat = Q

            if terminal:
                # apply discount to rewards
                r = np.full(len(rewards), discount_factor) ** np.arange(len(rewards)) * np.array(rewards)
                r = r[::-1].cumsum()[::-1]
                
                total_rewards.append(sum(rewards))

                # get running average of last 100 rewards, print every 100 episodes
                average_reward = np.mean(total_rewards[-100:])
                if episode % 10 == 0:

                    print(f"average of last 100 rewards as of episode {episode}: {average_reward:.2f}")

                if episode % 1000 == 0:
                    scheduler.step()
                # quit early if average_reward is high enough
                # if average_reward > early_stop:
                #     return total_rewards

                break

    return total_rewards



if __name__ == '__main__':
    # Parameters

    with open('data/dimension.pickle', 'rb') as pk:
        dim_info = pickle.load(pk)
    

    for user_class in range(10):
        Sl_d, Sf_d, A_d = 1, dim_info['tages'], dim_info['tages']
        T = 100
        SCALE = 0.05
        # user_class = 2
        model = Model(Sl_d, Sf_d, A_d, T, bias=0.3, normal_scale=SCALE)
        Q = Qnn(model.Sf_d+1)
        Q_hat = copy.deepcopy(Q)
        rewards = dqn(
            model=model, user_class=user_class, batch_size=10,
            Q=Q, Q_hat=Q_hat, num_episodes=200, learning_rate=0.01,
            learnign_rate_decay=1., e_greedy=0.02
        )

        with open(f'data/dqn_total_rewards_{user_class}.pickle', 'wb') as pk:
            pickle.dump(rewards, pk, protocol=pickle.HIGHEST_PROTOCOL)
        
from model import Model

import torch as tc
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import pickle
from tqdm import tqdm


# DEVICE = tc.device("cuda:1" if tc.cuda.is_available() else "cpu")

class Policy(nn.Module):
    def __init__(self, input_layer):
        super(Policy, self).__init__()
        self.net = nn.Sequential(nn.Linear(input_layer, input_layer), 
            nn.Sigmoid(), nn.Linear(input_layer, input_layer),
            nn.Softmax(dim=0))

        self.net.apply(self.weights_init)

        self.input_layer = input_layer

    def forward(self, x):
        x = tc.FloatTensor(x)
        return self.net(x)
        
    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            tc.nn.init.normal_(m.weight, mean=0.0, std=0.5)
            # tc.nn.init.zero_(m.bias)



def policy_gradient(model, user_class, policy, learning_rate=0.01, num_episodes=1000, batch_size=10, discount_factor=1., early_stop=2000):
    total_rewards, batch_rewards, batch_states_fast, batch_actions = [], [], [], []
    batch_counter = 1

    optimizer = tc.optim.Adam(policy.parameters(), lr=learning_rate)

    for episode in tqdm(range(num_episodes)):
        state_slow, state_fast = model.reset(user_class)
        rewards, actions, states_slow, states_fast = [], [], [], []

        while True:
            # use policy to make predictions and run an action
            # state_fast = tc.FloatTensor(state_fast).to(DEVICE)
            action_probs = policy(state_fast).detach().cpu().numpy()
            action = [np.argmax(action_probs)]
            action_vec = np.zeros(len(action_probs))
            action_vec[action] = 1

            # push all episodic data, move to next observation
            states_slow.append(state_slow)
            states_fast.append(state_fast)

            state_slow, state_fast, terminal = model.update(state_slow, state_fast, action_vec, user_class)
            reward = model.reward(state_slow, state_fast, action_vec)
            
            rewards.append(reward)
            actions.append(action)


            if terminal:
                # apply discount to rewards
                r = np.full(len(rewards), discount_factor) ** np.arange(len(rewards)) * np.array(rewards)
                r = r[::-1].cumsum()[::-1]
                discounted_rewards = r - r.mean()

                # collect the per-batch rewards, observations, actions
                batch_rewards.extend(discounted_rewards)
                batch_states_fast.extend(states_fast)
                batch_actions.extend(actions)
                batch_counter += 1
                total_rewards.append(sum(rewards))

                if batch_counter >= batch_size:
                    # reset gradient
                    optimizer.zero_grad()

                    # tensorify things
                    batch_rewards = tc.FloatTensor(batch_rewards)
                    batch_states_fast = tc.FloatTensor(batch_states_fast)
                    batch_actions = tc.LongTensor(batch_actions)
                    
                    # calculate loss
                    logprob = tc.log(policy(batch_states_fast))
                    batch_actions = batch_actions.reshape(len(batch_actions), 1)

                    # print(batch_states_fast.shape, logprob.shape, batch_actions.shape, batch_rewards.shape)
                    # break
                    selected_logprobs = batch_rewards * tc.gather(logprob, 1, batch_actions).squeeze()
                    loss = -selected_logprobs.mean()

                    # backprop/optimize
                    loss.backward()
                    optimizer.step()

                    # reset the batch
                    batch_rewards, batch_states_fast, batch_actions = [], [], []
                    batch_counter = 1

                # get running average of last 100 rewards, print every 100 episodes
                average_reward = np.mean(total_rewards[-100:])
                if episode % 10 == 0:
                    print(f"average of last 100 rewards as of episode {episode}: {average_reward:.2f}")

                # quit early if average_reward is high enough
                # if average_reward > early_stop:
                #     return total_rewards

                break

    return total_rewards



if __name__ == '__main__':
    # Parameters

    with open('data/dimension.pickle', 'rb') as pk:
        dim_info = pickle.load(pk)
    Sl_d, Sf_d, A_d = 1, dim_info['tages'], dim_info['tages']
    T = 100
    model = Model(Sl_d, Sf_d, A_d, T)
    policy = Policy(model.Sf_d)
    user_class = 3
    rewards = policy_gradient(
        model=model, user_class=user_class, batch_size=10,
        policy=policy, num_episodes=1500, learning_rate=0.01,
        early_stop=5
    )

    # moving average
    moving_average_num = 100
    def moving_average(x, n=moving_average_num):
        cumsum = np.cumsum(np.insert(x, 0, 0))
        return (cumsum[n:] - cumsum[:-n]) / float(n)

    # plotting
    
    fig, ax = plt.subplots()
    ax.scatter(np.arange(len(rewards)), rewards, label='episodes', alpha=0.5)
    ax.plot(moving_average(rewards), label=f'average')
    ax.set_title(f'Policy Gradient')
    ax.set_label('Episode')
    ax.set_label('Reward')
    plt.savefig("MC.png", dpi=200)

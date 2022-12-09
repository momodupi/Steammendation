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
            nn.ReLU(), nn.Linear(input_layer, input_layer),
            nn.Softmax(dim=-1))
            # nn.ReLU6())
        # self.net.to(DEVICE)
        self.input_layer = input_layer

    def forward(self, x):
        x = tc.FloatTensor(x)
        return self.net(x)


def policy_gradient(model, user_class, policy, learning_rate=0.01, num_episodes=1500, batch_size=10, discount_factor=0.99, early_stop_reward=100):
    total_rewards, batch_rewards, batch_states_fast, batch_actions = [], [], [], []
    batch_counter = 1

    optimizer = tc.optim.Adam(policy.parameters(), lr=learning_rate)
    # action_space = np.arange(env.action_space.n) # [0, 1] for cartpole (either left or right)

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
                if episode % 100 == 0:
                    print(f"average of last 100 rewards as of episode {episode}: {average_reward:.2f}")

                # quit early if average_reward is high enough
                # if average_reward > early_stop_reward:
                #     return total_rewards

                break

    return total_rewards



if __name__ == '__main__':
    # Parameters
    num_episode = 30
    batch_size = 5
    learning_rate = 0.01
    # gamma = 1

    with open('data/dimension.pickle', 'rb') as pk:
        dim_info = pickle.load(pk)
    Sl_d, Sf_d, A_d = 1, dim_info['tages'], dim_info['tages']
    T = 100
    model = Model(Sl_d, Sf_d, A_d, T)
    policy = Policy(model.Sf_d)
    user_class = 1
    rewards = policy_gradient(
        model=model, user_class=user_class,
        policy = policy, num_episodes=1500
    )

    # moving average
    moving_average_num = 100
    def moving_average(x, n=moving_average_num):
        cumsum = np.cumsum(np.insert(x, 0, 0))
        return (cumsum[n:] - cumsum[:-n]) / float(n)

    # plotting
    plt.scatter(np.arange(len(rewards)), rewards, label='individual episodes')
    plt.plot(moving_average(rewards), label=f'moving average of last {moving_average_num} episodes')
    plt.title(f'Vanilla Policy Gradient')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.show()

    # reward_history = []
    # policy_error = []

    # user_class = 1

    # policy = Policy(model.Sf_d)
    # optimizer = tc.optim.Adam(policy.parameters(), lr=learning_rate)

    # # Batch History
    # state_slow_pool = []
    # state_fast_pool = []
    # action_pool = []
    # reward_pool = []

    # for _ in tqdm(range(num_episode)):
    # # for _ in range(num_episode):
    #     state_slow, state_fast = model.reset(user_class)
    #     step = 0
    #     # MC
    #     for t in range(150):
    #         state_fast = Variable(state_fast)
    #         action = policy(state_fast)
            
    #         reward = model.reward(state_slow, state_fast, action)
    #         next_state_slow, next_state_fast, terminal = model.update(state_slow, state_fast, action, user_class)
            
    #         state_slow_pool.append(state_slow)
    #         state_fast_pool.append(state_fast)
    #         action_pool.append(action)
    #         reward_pool.append(reward)
            
    #         state_slow = next_state_slow
    #         state_fast = next_state_fast
    #         step += 1
    #         if terminal:
    #             break
    #     # print(f'pool: {_},', reward_pool, state_slow_pool, action_pool)
        
    #     G = np.cumsum( np.array(reward_pool[::-1]) )
    #     reward_history.append(G[-1])
    
    #     for i in range(step):
    #         state_fast = state_fast_pool[i]
    #         action = Variable(action_pool[i])
    #         loss = - tc.sum(tc.log( policy(state_fast) ) * G[i])

    #         # Gradient Desent
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()

    #     state_slow_pool = []
    #     state_fast_pool = []
    #     action_pool = []
    #     reward_pool = []


    # print(reward_history)
    # print(model.y_0[user_class], policy(tc.tensor(model.y_0[user_class], dtype=tc.float)))
    
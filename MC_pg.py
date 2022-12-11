from model import Model

import torch as tc
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import pickle
from tqdm import tqdm


DEVICE = tc.device("cuda:1" if tc.cuda.is_available() else "cpu")

class Policy(nn.Module):
    def __init__(self, input_layer):
        super(Policy, self).__init__()
        self.net = nn.Sequential(nn.Linear(input_layer, 2*input_layer), 
            nn.Sigmoid(), nn.Linear(2*input_layer, input_layer),
            nn.Softmax(dim=0)).to(DEVICE)

        self.net.apply(self.weights_init)

        self.input_layer = input_layer

    def forward(self, x):
        x = tc.tensor(x, dtype=tc.float, device=DEVICE)
        return self.net(x)
        
    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            tc.nn.init.normal_(m.weight, mean=0.0, std=0.5)
            # tc.nn.init.zero_(m.bias)



def MC_policy_gradient(model, user_class, policy, learning_rate=0.01, num_episodes=1000, batch_size=10, discount_factor=1., learnign_rate_decay=0.9):
    total_rewards, batch_rewards, batch_states_fast, batch_actions = [], [], [], []
    batch_counter = 1

    optimizer = tc.optim.Adam(policy.parameters(), lr=learning_rate)
    scheduler = tc.optim.lr_scheduler.ExponentialLR(optimizer, gamma=learnign_rate_decay)

    for episode in tqdm(range(num_episodes)):
        state_slow, state_fast = model.reset(user_class)
        rewards, actions, states_slow, states_fast = [], [], [], []

        while True:
            # use policy to make predictions and run an action
            # state_fast = tc.FloatTensor(state_fast).to(DEVICE)
            action_probs = policy(state_fast).detach().cpu().numpy()
            # action = [np.argmax(action_probs)]
            action = [ np.random.choice(len(action_probs), 1, p=action_probs) ]
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
                    batch_rewards = tc.tensor(batch_rewards, dtype=tc.float, device=DEVICE)
                    batch_states_fast = tc.tensor(batch_states_fast, dtype=tc.float, device=DEVICE)
                    batch_actions = tc.tensor(batch_actions, dtype=tc.int64, device=DEVICE)
                    
                    # calculate loss
                    logprob = tc.log(policy(batch_states_fast))
                    batch_actions = batch_actions.reshape(len(batch_actions), 1)

                    # print(batch_states_fast, batch_actions, batch_rewards)
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
                    scheduler.step()
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
    
    user_class = 4
    model = Model(Sl_d, Sf_d, A_d, T, bias=0.3)
    policy = Policy(model.Sf_d)
    rewards = MC_policy_gradient(
        model=model, user_class=user_class, batch_size=30,
        policy=policy, num_episodes=500, learning_rate=0.3,
        learnign_rate_decay=0.9
    )

    with open(f'data/pg_total_rewards_{user_class}.pickle', 'wb') as pk:
        pickle.dump(rewards, pk, protocol=pickle.HIGHEST_PROTOCOL)
    
    
    # for user_class in range(10):
    # # user_class = 8
    #     model = Model(Sl_d, Sf_d, A_d, T, bias=0.3)
    #     policy = Policy(model.Sf_d)
    #     rewards = policy_gradient(
    #         model=model, user_class=user_class, batch_size=5,
    #         policy=policy, num_episodes=500, learning_rate=0.01,
    #         learnign_rate_decay=0.9
    #     )

    #     with open(f'data/pg_total_rewards_{user_class}.pickle', 'wb') as pk:
    #         pickle.dump(rewards, pk, protocol=pickle.HIGHEST_PROTOCOL)

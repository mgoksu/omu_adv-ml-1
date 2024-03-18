
import random

import torch
import gym
from torch.autograd import Variable
import matplotlib.pyplot as plt


from network import Network

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor

# Hyperparameters
HIDDEN_LAYER = 32 
INPUT_SIZE = 4
OUTPUT_SIZE = 2
N_EPISODES = 500
LR = 0.01
GAMMA = 0.99

ENV = gym.make('CartPole-v0').unwrapped

model = Network(INPUT_SIZE, HIDDEN_LAYER, OUTPUT_SIZE)
if use_cuda:
    model.cuda()

optim = torch.optim.Adam(model.parameters(), lr=LR)


def discount_rewards(r):
    discounted_r = torch.zeros(r.size())
    running_add = 0
    for t in reversed(range(len(r))):
        running_add = running_add * GAMMA + r[t]
        discounted_r[t] = running_add

    return discounted_r

def run_episode(net, e, env):
    state = env.reset()
    reward_sum = 0
    xs = FloatTensor([])
    ys = FloatTensor([])
    rewards = FloatTensor([])
    steps = 0

    while True:

        x = FloatTensor([state])
        xs = torch.cat([xs, x])

        action_prob = net(Variable(x))

        # select an action depending on the probability
        action = 0 if random.random() < action_prob.data[0][0] else 1

        y = FloatTensor([[1, 0]] if action == 0 else [[0, 1]])
        ys = torch.cat([ys, y])

        state, reward, done, _ = env.step(action)
        rewards = torch.cat([rewards, FloatTensor([[reward]])])
        reward_sum += reward
        steps += 1

        if done or steps >= 500:
            adv = discount_rewards(rewards)
            adv = (adv - adv.mean())/(adv.std() + 1e-7)
            loss = learn(xs, ys, adv)
            print("[Episode {:>5}]  steps: {:>5} loss: {:>5}".format(e, steps, loss))
            return reward_sum

def learn(x, y, adv):
    # Loss function, ∑ Ai*logp(yi∣xi), but we need fake label Y due to autodiff
    action_pred = model(Variable(x))
    y = Variable(y, requires_grad=True)
    adv = Variable(adv)

    log_lik = -y * torch.log(action_pred)

    log_lik_adv = log_lik * adv

    loss = torch.sum(log_lik_adv, 1).mean()

    optim.zero_grad()
    loss.backward()
    optim.step()

    return loss.item()

def is_solved(history):
    # condition for success given by https://github.com/openai/gym/wiki/CartPole-v0
    return len(history) > 100 and sum(history[-100:])/100 > 195

def plot_durations(d):
    plt.figure(2)
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(d)

    plt.savefig('duration_scores.png')


if __name__ == '__main__':
    # train
    history = []
    for e in range(N_EPISODES):
        episode_reward_sum = run_episode(model, e, ENV)
        history.append(episode_reward_sum)

        finish_condition = is_solved(history)

        if finish_condition:
            print('Training is finished!')
            break

    # save model
    torch.save(model, 'model.pt')

    plot_durations(history)

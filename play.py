
import torch
from torch.autograd import Variable
from moviepy.editor import ImageSequenceClip

import gym

ENV = gym.make('CartPole-v0').unwrapped

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor


# load model
model = torch.load('model.pt')

def play(env):
    state = env.reset()
    steps = 0
    frames = []
    while True:
        frame = env.render(mode='rgb_array')
        frames.append(frame)
        action = torch.max(model(Variable(FloatTensor([state]))), 1)[1].data[0]
        next_state, _, done, _ = env.step(action.item())

        state = next_state
        steps += 1

        if done or steps >= 2000:
            break

    clip = ImageSequenceClip(frames, fps=100)
    clip.write_videofile('play.mp4', fps=100)

if __name__ == '__main__':
    play(ENV)

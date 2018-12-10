from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from gym import wrappers


def record_video(agent, env, n_episodes=1, filename="video"):
    env = wrappers.Monitor(env, filename)

    for episode in range(n_episodes):
        s = env.reset()
        done = False
        while not done:
            env.render()
            a = agent.choose_action(s)
            s, r, done, _ = env.step(a)
    env.close()


def visualize(agent, env, n_episodes=1):
    for episode in range(n_episodes):
        s = env.reset()
        done = False
        while not done:
            env.render()
            a = agent.choose_action(s)
            s, r, done, _ = env.step(a)
    env.close()

'''''''''
@file: run_MountainCar_Example.py
@author: MRL Liu
@time: 2021/3/12 11:20
@env: Python,Numpy
@desc:
@ref:
@blog: https://blog.csdn.net/qq_41959920
'''''''''

import gym
import matplotlib.pyplot as plt
from Policy_gradient.REINFORCE.Agent_REINFORCE import Agent_REINFORCE


DISPLAY_REWARD_THRESHOLD = -2000  # renders environment if total episode reward is greater then this threshold
# episode: 154   reward: -10667
# episode: 387   reward: -2009
# episode: 489   reward: -1006
# episode: 628   reward: -502
TEST_ENV_NAME ='MountainCar-v0'

def train(agent,num_episode=1000,is_Render = False):
    for i_episode in range(num_episode):
        # running_reward = 0
        observation = env.reset()

        while True:
            if is_Render: env.render()

            action = agent.choose_action(observation)

            observation_, reward, done, info = env.step(action)

            agent.store_in_memory(observation, action, reward)

            if done:
                ep_rs_sum = sum(agent.ep_rs)

                if 'running_reward' not in globals():
                    running_reward = ep_rs_sum
                else:
                    running_reward = running_reward * 0.99 + ep_rs_sum * 0.01
                #if running_reward > DISPLAY_REWARD_THRESHOLD: is_Render = True  # rendering
                print("episode:", i_episode, "  reward:", int(running_reward))

                vt = agent.learn()

                if i_episode == 30:
                    plt.plot(vt)  # plot the episode vt
                    plt.xlabel('episode steps')
                    plt.ylabel('normalized state-action value')
                    plt.show()
                break

            observation = observation_

if __name__=='__main__':
    env = gym.make(TEST_ENV_NAME)
    env = env.unwrapped
    env.seed(1)

    print(env.action_space)
    print(env.observation_space)
    print(env.observation_space.high)
    print(env.observation_space.low)

    STATE_FEATURES = env.observation_space.shape[0]
    ACTION_SPACE = env.action_space.n

    agent = Agent_REINFORCE(
        n_features=STATE_FEATURES,
        n_actions=ACTION_SPACE,
        learning_rate=0.02,
        reward_decay=0.995,
        output_graph=False,
    )

    train(agent,num_episode=1000)
'''''''''
@file: run_MountainCar_Example.py
@author: MRL Liu
@time: 2021/3/10 14:51
@env: Python,Numpy
@desc:
@ref:
@blog: https://blog.csdn.net/qq_41959920
'''''''''
import gym
from DQN.Prioritized_Replay_DQN_on_Gym.Agent_Prioritized_Replay_DQN import Agent_Prioritized_Replay_DQN
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


MEMORY_SIZE = 10000


def train(RL):
    total_steps = 0
    steps = []
    episodes = []
    for i_episode in range(20):
        observation = env.reset()
        while True:
            # env.render()

            action = RL.choose_action(observation)

            observation_, reward, done, info = env.step(action)

            if done: reward = 10

            RL.store_in_memory(observation, action, reward, observation_)

            if total_steps > MEMORY_SIZE:
                RL.learn()

            if done:
                print('episode ', i_episode, ' finished')
                steps.append(total_steps)# 记录步数
                episodes.append(i_episode)# 记录回合数
                break

            observation = observation_
            total_steps += 1
    return np.vstack((episodes, steps))



if __name__=='__main__':
    env = gym.make('MountainCar-v0')
    env = env.unwrapped
    env.seed(21)

    sess = tf.Session()
    with tf.variable_scope('natural_DQN'):
        RL_natural = Agent_Prioritized_Replay_DQN(
            n_actions=3,
            n_features=2,
            memory_size=MEMORY_SIZE,
            e_greedy_increment=0.00005,
            sess=sess,
            prioritized=False,
        )

    with tf.variable_scope('DQN_with_prioritized_replay'):
        RL_prio = Agent_Prioritized_Replay_DQN(
            n_actions=3,
            n_features=2,
            memory_size=MEMORY_SIZE,
            e_greedy_increment=0.00005,
            sess=sess,
            prioritized=True,
            output_graph=True,
        )
    sess.run(tf.global_variables_initializer())

    his_natural = train(RL_natural)
    his_prio = train(RL_prio)

    # compare based on first success
    plt.plot(his_natural[0, :], his_natural[1, :] - his_natural[1, 0], c='b', label='natural DQN')
    plt.plot(his_prio[0, :], his_prio[1, :] - his_prio[1, 0], c='r', label='DQN with prioritized replay')
    plt.legend(loc='best')
    plt.ylabel('total training time')
    plt.xlabel('episode')
    plt.grid()
    plt.show()
'''''''''
@file: run_Pendulum_Example.py
@author: MRL Liu
@time: 2021/3/10 13:15
@env: Python,Numpy
@desc:
@ref:
@blog: https://blog.csdn.net/qq_41959920
'''''''''
import gym
from DQN.Dueling_DQN_on_Gym.Agent_Dueling_DQN import Agent_Dueling_DQN
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

MEMORY_SIZE = 3000
ACTION_SPACE = 25


def train(RL):
    acc_r = [0]
    total_steps = 0
    observation = env.reset()
    while True:
        # if total_steps-MEMORY_SIZE > 9000: env.render()

        action = RL.choose_action(observation)

        f_action = (action - (ACTION_SPACE - 1) / 2) / ((ACTION_SPACE - 1) / 4)  # [-2 ~ 2] float actions
        observation_, reward, done, info = env.step(np.array([f_action]))

        reward /= 10  # normalize to a range of (-1, 0)
        acc_r.append(reward + acc_r[-1])  # accumulated reward

        RL.store_in_memory(observation, action, reward, observation_)

        if total_steps > MEMORY_SIZE:
            RL.learn()

        if total_steps - MEMORY_SIZE > 15000:
            break

        observation = observation_
        total_steps += 1
    return RL.cost_his, acc_r


if __name__=="__main__":
    env = gym.make('Pendulum-v0')
    env = env.unwrapped
    env.seed(1)
    sess = tf.Session()
    with tf.variable_scope('Natural_DQN'):
        natural_DQN = Agent_Dueling_DQN(
            n_actions=ACTION_SPACE,
            n_features=3,
            memory_size=MEMORY_SIZE,
            e_greedy_increment=0.001,
            sess=sess,
            dueling=False)

    with tf.variable_scope('dueling'):
        dueling_DQN = Agent_Dueling_DQN(
            n_actions=ACTION_SPACE,
            n_features=3,
            memory_size=MEMORY_SIZE,
            e_greedy_increment=0.001,
            sess=sess,
            dueling=True,
            output_graph=True)

    sess.run(tf.global_variables_initializer())


    #c_natural, r_natural = train(natural_DQN)
    c_dueling, r_dueling = train(dueling_DQN)

    plt.figure(1)
    #plt.plot(np.array(c_natural), c='r', label='natural')
    plt.plot(np.array(c_dueling), c='b', label='dueling')
    plt.legend(loc='best')
    plt.ylabel('cost')
    plt.xlabel('training steps')
    plt.grid()

    plt.figure(2)
    #plt.plot(np.array(r_natural), c='r', label='natural')
    plt.plot(np.array(r_dueling), c='b', label='dueling')
    plt.legend(loc='best')
    plt.ylabel('accumulated reward')
    plt.xlabel('training steps')
    plt.grid()

    plt.show()

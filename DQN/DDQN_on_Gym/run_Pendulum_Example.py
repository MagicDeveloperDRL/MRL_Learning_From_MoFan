import gym
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from DQN.DDQN_on_Gym.Agent_DDQN import Agent_DDQN

MEMORY_SIZE = 3000
ACTION_SPACE = 11

# 训练一个回合
def train(agent):
    total_step = 0
    observation = env.reset()
    while True:
        action = agent.choose_Action(observation)

        f_action = (action - (ACTION_SPACE - 1) / 2) / ((ACTION_SPACE - 1) / 4)  # convert to [-2 ~ 2] float actions
        observation_, reward, done, info = env.step(np.array([f_action]))
        reward /= 10

        agent.store_in_memory(observation, action, reward, observation_)

        if total_step > MEMORY_SIZE:
            agent.learn()

        if done or total_step - MEMORY_SIZE > 20000:
            break
        observation = observation_
        total_step += 1
    return agent.q



if __name__=="__main__":
    env = gym.make('Pendulum-v0')
    env = env.unwrapped
    env.seed(1)
    sess = tf.Session()
    with tf.variable_scope('Natural_DQN'):
        natural_DQN = Agent_DDQN(
            n_features=3,
            n_actions=ACTION_SPACE,
            memory_size=MEMORY_SIZE,
            e_greedy_increment=0.001,
            double_q=False,
            sess=sess
        )
    with tf.variable_scope('Double_DQN'):
        double_DQN = Agent_DDQN(
            n_features=3,
            n_actions=ACTION_SPACE,
            memory_size=MEMORY_SIZE,
            e_greedy_increment=0.001,
            double_q=True,
            sess=sess,
            output_graph=True
        )
    sess.run(tf.global_variables_initializer())

    q_natural = train(natural_DQN)
    q_double =train(double_DQN)



    plt.plot(np.array(q_natural), c='r', label='natural')
    plt.plot(np.array(q_double), c='b', label='double')
    plt.legend(loc='best')
    plt.ylabel('Q eval')
    plt.xlabel('training steps')
    plt.grid()
    plt.show()
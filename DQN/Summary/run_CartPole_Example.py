'''''''''
@file: run_this.py
@author: MRL Liu
@time: 2021/3/10 21:33
@env: Python,Numpy
@desc:
@ref:
@blog: https://blog.csdn.net/qq_41959920
'''''''''
import gym
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from DQN.Summary.Agent_DQN import Agent_DQN
from DQN.Summary.Agent_DDQN import Agent_DDQN
from DQN.Summary.Agent_Dueling_DQN import Agent_Dueling_DQN
from DQN.Summary.Agent_Prioritized_Replay_DQN import Agent_Prioritized_Replay_DQN

MEMORY_SIZE = 3000
ACTION_SPACE = 11
STATE_FEATURES = 3
TEST_ENV_NAME ='CartPole-v0'


# 训练一个回合
def train_one_step(agent):
    reward_his = [0] # 精度列表
    total_step = 0
    # 获取初始化状态
    observation = env.reset()
    while True:
        # 刷新环境
        #env.render()
        # 从环境中获取反馈
        action = agent.choose_action(observation)
        # 从环境中获取反馈
        observation_, reward, done, info = env.step(action)
        x, x_dot, theta, theta_dot = observation_
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        reward = r1 + r2
        reward_his.append(reward + reward_his[-1])  # 评估的奖励
        # 存储到内存中
        agent.store_in_memory(observation, action, reward, observation_)
        # 是否开始学习
        if total_step > MEMORY_SIZE:
            agent.learn()
        # 更新状态和计数器
        observation = observation_
        total_step += 1
        # 是否结束本回合仿真
        if done or total_step - MEMORY_SIZE > 20000:
            print('本回合训练结束，总共训练了{}个steps'.format(total_step))
            break

    return agent.q_his,reward_his,agent.cost_his

def plot_Q_eval():
    fig = plt.figure()
    fig.canvas.set_window_title(TEST_ENV_NAME+' Q eval History')
    plt.plot(np.array(q_natural), c='r', label='natural')
    plt.plot(np.array(q_double), c='b', label='double')
    plt.plot(np.array(q_dueling), c='g', label='dueling')
    plt.plot(np.array(q_prioritized), c='y', label='prioritized')
    plt.legend(loc='best')
    plt.ylabel('Q eval')
    plt.xlabel('training steps')
    plt.grid()
    plt.show()
def plot_Reward():
    fig = plt.figure()
    fig.canvas.set_window_title(TEST_ENV_NAME+' Reward History')
    plt.plot(np.array(r_natural), c='r', label='natural')
    plt.plot(np.array(r_double), c='b', label='double')
    plt.plot(np.array(r_dueling), c='g', label='dueling')
    plt.plot(np.array(r_prioritized), c='y', label='prioritized')
    plt.legend(loc='best')
    plt.ylabel('accumulated reward')
    plt.xlabel('training steps')
    plt.grid()
    plt.show()
def plot_Cost():
    fig = plt.figure()
    fig.canvas.set_window_title(TEST_ENV_NAME+' Cost History')
    plt.plot(np.array(c_natural), c='r', label='natural')
    plt.plot(np.array(c_double), c='b', label='double')
    plt.plot(np.array(c_dueling), c='g', label='dueling')
    plt.plot(np.array(c_prioritized), c='y', label='prioritized')
    plt.legend(loc='best')
    plt.ylabel('Cost')
    plt.xlabel('training steps')
    plt.grid()
    plt.show()

if __name__=="__main__":
    # 导入Gym库中的一个环境
    env = gym.make(TEST_ENV_NAME)
    env = env.unwrapped  # 不做这个会有很多限制
    env.seed(1)

    ACTION_SPACE = env.action_space.n
    STATE_FEATURES = env.observation_space.shape[0]

    sess = tf.Session()
    with tf.variable_scope('Natural_DQN'):
        natural_DQN = Agent_DQN(
            n_features=STATE_FEATURES,
            n_actions=ACTION_SPACE,
            memory_size=MEMORY_SIZE,
            e_greedy_increment=0.001,
            sess=sess,
            output_graph=False
        )
    with tf.variable_scope('Double_DQN'):
        double_DQN = Agent_DDQN(
            n_features=STATE_FEATURES,
            n_actions=ACTION_SPACE,
            memory_size=MEMORY_SIZE,
            e_greedy_increment=0.001,
            sess=sess,
            output_graph=False
        )
    with tf.variable_scope('Dueling_DQN'):
        dueling_DQN = Agent_Dueling_DQN(
            n_features=STATE_FEATURES,
            n_actions=ACTION_SPACE,
            memory_size=MEMORY_SIZE,
            e_greedy_increment=0.001,
            sess=sess,
            output_graph=False
        )
    with tf.variable_scope('Prioritized_DQN'):
        prioritized_DQN = Agent_Prioritized_Replay_DQN(
            n_features=STATE_FEATURES,
            n_actions=ACTION_SPACE,
            memory_size=MEMORY_SIZE,
            e_greedy_increment=0.001,
            sess=sess,
            prioritized=True,
            output_graph=False
        )
    sess.run(tf.global_variables_initializer())

    q_natural, r_natural, c_natural = train_one_step(natural_DQN)
    q_double, r_double, c_double = train_one_step(double_DQN)
    q_dueling, r_dueling, c_dueling = train_one_step(dueling_DQN)
    q_prioritized, r_prioritized, c_prioritized = train_one_step(prioritized_DQN)

    plot_Q_eval()
    plot_Reward()
    plot_Cost()
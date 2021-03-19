'''''''''
@file: run_Pendulum_Example.py
@author: MRL Liu
@time: 2021/3/18 19:53
@env: Python,Numpy
@desc:
@ref:
@blog: https://blog.csdn.net/qq_41959920
'''''''''
import tensorflow as tf
import numpy as np
import gym
import time
from Policy_gradient.DDPG.Agent_DDPG import Agent_DDPG

np.random.seed(1)
tf.set_random_seed(1)

# 超参数
MAX_EPISODES = 200
MAX_EP_STEPS = 200
LR_A = 0.001    # actor的学习率
LR_C = 0.002    # critic的学习率
GAMMA = 0.9     # 奖励折扣
REPLACEMENT = [
    dict(name='soft', tau=0.01),
    dict(name='hard', rep_iter_a=600, rep_iter_c=500)
][0]            # you can try different target replacement strategies
MEMORY_CAPACITY = 10000 # 记忆池的最大容量
BATCH_SIZE = 32

RENDER = False
OUTPUT_GRAPH = False
TEST_ENV_NAME = 'Pendulum-v0'

def train(max_episodes=3000,max_steps_ep=1000,is_Render = False):
    var = 3  # control exploration
    t1 = time.time() # 记录时间
    for i in range(max_episodes):
        ep_reward = 0
        s = env.reset() # 初始化环境状态
        for j in range(max_steps_ep):

            if is_Render:
                env.render()

            # Add exploration noise
            a = agent.choose_action(s) # 选择动作
            a = np.clip(np.random.normal(a, var), -2, 2)  # add randomness to action selection for exploration
            s_, r, done, info = env.step(a) # 获取环境反馈

            agent.store_transition(s, a, r / 10, s_) # 存储到经验池中
            if agent.memory.pointer > MEMORY_CAPACITY:
                var *= .9995  # decay the action randomness
                agent.learn()
            # 切换观察值
            s = s_
            ep_reward += r

            if done or j == max_steps_ep - 1 :
                print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, )
                if ep_reward > -300:
                    is_Render = True
                break

    print('Running time: ', time.time() - t1)

if __name__ == '__main__':
    # 创建环境
    env = gym.make(TEST_ENV_NAME)
    env = env.unwrapped
    env.seed(1)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high

    # 创建Agent
    sess = tf.Session()
    agent = Agent_DDPG(
        state_dim=state_dim,
        action_dim=action_dim,
        action_bound=action_bound,
        a_lr=LR_A,
        c_lr=LR_C,
        memory_size=MEMORY_CAPACITY,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        replacement=REPLACEMENT,
        sess=sess)

    sess.run(tf.global_variables_initializer())

    if OUTPUT_GRAPH:
        tf.summary.FileWriter("logs/", sess.graph)

    train(MAX_EPISODES,MAX_EP_STEPS,RENDER)




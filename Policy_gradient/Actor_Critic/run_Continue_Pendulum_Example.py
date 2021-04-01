'''''''''
@file: run_Continue_Pendulum_Example.py
@author: MRL Liu
@time: 2021/3/18 12:25
@env: Python,Numpy
@desc:本模块用来处理连续动作空间的任务
@ref:
@blog: https://blog.csdn.net/qq_41959920
'''''''''
import tensorflow as tf
import numpy as np
import gym
from MRL_Learning_From_MoFan.Policy_gradient.Actor_Critic_frame.Agent_AC import Actor_Continue,Critic_Continue

np.random.seed(2)
tf.set_random_seed(2)
# 超参数
OUTPUT_GRAPH = False # 是否输出计算图
MAX_EPISODE = 1000 # 最大训练数量
MAX_EP_STEPS = 200 # 每回合的最大训练数
DISPLAY_REWARD_THRESHOLD = -100  # 如果累计奖励超过该数就开始渲染环境
RENDER = False  # 是否渲染环境
GAMMA = 0.9     # TD error中的reward discount
LR_A = 0.001    # actor的学习率
LR_C = 0.01     # critic的学习率，比actor更快
TEST_ENV_NAME ='Pendulum-v0' # 测试环境名称

# 训练
def train(num_episode,is_Render=False):
    for i_episode in range(num_episode):
        t_step = 0 # step计数器
        track_r = []
        s = env.reset()  # 初始化环境状态
        while True:
            if is_Render:
                env.render()
            a = actor.choose_action(s)
            s_, r, done, info = env.step(a)
            r /= 10
            track_r.append(r)
            td_error = critic.learn(s, r, s_)  # gradient = grad[r + gamma * V(s_) - V(s)]
            actor.learn(s, a, td_error)  # true_gradient = grad[logPi(s,a) * td_error]
            # 切换状态
            s = s_
            t_step += 1
            # 如果游戏失败或者超过最大step数
            if done or t_step > MAX_EP_STEPS:
                ep_rs_sum = sum(track_r)
                if 'running_reward' not in globals():
                    running_reward = ep_rs_sum
                else:
                    running_reward = running_reward * 0.9 + ep_rs_sum * 0.1
                if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True  # rendering
                print("episode:", i_episode, "  reward:", int(running_reward))
                break

if __name__=='__main__':
    # 创建环境
    env = gym.make(TEST_ENV_NAME)
    env.seed(1)
    env = env.unwrapped

    N_S = env.observation_space.shape[0]
    A_BOUND = env.action_space.high # 获取动作空间的最大值
    # 创建actor及critic
    sess = tf.Session()

    actor = Actor_Continue(sess,
                  n_features=N_S,
                  lr=LR_A,
                  action_bound=[-A_BOUND, A_BOUND])
    critic = Critic_Continue(sess,
                    n_features=N_S,
                    lr=LR_C)

    sess.run(tf.global_variables_initializer())

    if OUTPUT_GRAPH:
        tf.summary.FileWriter("logs/", sess.graph)
    # 仿真训练
    train(MAX_EPISODE, RENDER)
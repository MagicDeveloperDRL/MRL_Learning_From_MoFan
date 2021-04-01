'''''''''
@file: run_Pendulum_Example.py
@author: MRL Liu
@time: 2021/3/28 17:23
@env: Python,Numpy
@desc: 
@ref:
@blog: https://blog.csdn.net/qq_41959920
'''''''''
import gym
import tensorflow as tf
import multiprocessing
import numpy as np
import os
import threading
import shutil
import matplotlib.pyplot as plt
from Policy_gradient.A3C.Agent_A3C import ACNet,Worker

# 代码配置
TEST_ENV_NAME = 'Pendulum-v0' # 测试环境
OUTPUT_GRAPH = True # 是否输出计算图
LOG_DIR = './log' # 计算图输出目录
N_WORKERS = multiprocessing.cpu_count() # 获取cpu的数量

# 超参数
LR_A = 0.0001    # learning rate for actor
LR_C = 0.001    # learning rate for critic

if __name__=='__main__':
    # 创建环境
    env = gym.make(TEST_ENV_NAME) # 定义使用gym中的某一个环境
    env = env.unwrapped # 设置环境打开其特殊限制
    env.seed(1) # 设置环境的随机种子

    N_S = env.observation_space.shape[0] # 获取环境的状态空间的特征数
    N_A = env.action_space.shape[0] # 获取环境的动作空间的个数
    A_BOUND = [env.action_space.low, env.action_space.high] # 获取连续动作的限制范围

    SESS = tf.Session()

    with tf.device("/cpu:0"):  # 指定运行在第一块CPU上
        OPT_A = tf.train.RMSPropOptimizer(LR_A, name='RMSPropA')
        OPT_C = tf.train.RMSPropOptimizer(LR_C, name='RMSPropC')
        GLOBAL_AC = ACNet(GLOBAL_NET_SCOPE)  # we only need its params
        workers = []
        # 创建worker
        for i in range(N_WORKERS):
            i_name = 'W_%i' % i  # worker name
            workers.append(Worker(i_name, GLOBAL_AC))

    COORD = tf.train.Coordinator()  # 创建一个TensorFlow的线程管理器
    SESS.run(tf.global_variables_initializer())

    if OUTPUT_GRAPH:
        if os.path.exists(LOG_DIR):
            shutil.rmtree(LOG_DIR)
        tf.summary.FileWriter(LOG_DIR, SESS.graph)

    worker_threads = []  # 创建线程队列
    for worker in workers:
        job = lambda: worker.work()
        t = threading.Thread(target=job)  # 创建一个线程
        t.start()  # 让线程开始工作
        worker_threads.append(t)  # 将线程添加到线程队列
    COORD.join(worker_threads)  # 将线程加入到tf的线程管理器

    plt.plot(np.arange(len(GLOBAL_RUNNING_R)), GLOBAL_RUNNING_R)
    plt.xlabel('step')
    plt.ylabel('Total moving reward')
    plt.show()
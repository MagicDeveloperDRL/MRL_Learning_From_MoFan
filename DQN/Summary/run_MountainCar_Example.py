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
from DQN.DQN_on_Maze.Agent_DQN import  Agent_DQN

# 在Gym环境下运行DQN算法的训练算法
def run_DQN_on_Gym():
    total_steps=0 # 记录步数
    # 进行100回合的训练
    for i_episode in range(10):
        observation = env.reset()
        ep_r = 0
        # 走多少步
        while True:
            # 刷新环境
            env.render()
            # 基于观测值选择动作
            action= agent.choose_Action(observation)
            # 获取环境的反馈
            observation_,reward,done,info= env.step(action)
            position, velocity = observation_
            # the higher the better
            reward = abs(position - (-0.5))  # r in [0, 1]
            # 将转换元组存入记忆池
            agent.store_in_memory(observation,action,reward,observation_)
            # 以固定频率进行学习
            if(total_steps>1000):
                agent.learn_from_step()
            ep_r += reward
            if done:
                get = '| Get' if observation_[0] >= env.unwrapped.goal_position else '| ----'
                print('Episode:', i_episode,
                      get,
                      '|Ep_r', round(ep_r, 4),
                      '|Epsilon:', round(agent.epsilon, 2))
                break
            # 切换观察值
            observation=observation_
            total_steps+=1



if __name__=="__main__":
    #导入Gym库中的一个环境
    env=gym.make('MountainCar-v0')
    env=env.unwrapped

    print(env.action_space)
    print(env.observation_space)
    print(env.observation_space.high)
    print(env.observation_space.low)
    # 创建Agent模块
    agent= Agent_DQN(n_actions=3,
                     n_features=2,
                     learning_rate=0.001,
                     e_greedy=0.9,
                     replace_target_iter=300,
                     memory_size=3000,
                     e_greedy_increment=0.0002
                     )
    # 开始训练算法
    run_DQN_on_Gym()
    # 最后输出cost曲线
    agent.plot_cost()
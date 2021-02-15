"""
@file: Run_this.py
@author: MRL Liu
@time: 2021/2/16 0:27
@env: Python,Numpy
@desc: 运行脚本
@ref:
@blog: https://blog.csdn.net/qq_41959920
"""
from Q_Learning.Summary.Maze_Env import Maze_Env
from Q_Learning.Summary.Line_Env import Line_Env

from Q_Learning.Summary.Agent import Line_Agent_Q_Learning
from Q_Learning.Summary.Agent import Maze_Agent_Q_Learning
from Q_Learning.Summary.Agent import Maze_Agent_Sarsa
from Q_Learning.Summary.Agent import Maze_Agent_Sarsa_Lambda

from Q_Learning.Summary.Trainer import Line_Trainer_Q_Learning
from Q_Learning.Summary.Trainer import Maze_Trainer_Q_Learning
from Q_Learning.Summary.Trainer import Maze_Trainer_Sarsa

def run_line():
    env = Line_Env()  # 创建环境
    agent = Line_Agent_Q_Learning()  # 创建agent
    trainer = Line_Trainer_Q_Learning(env, agent)  # 创建训练器
    # 训练agent模型
    trainer.train(max_episodes=10)
    # 使用agent模型
    trainer.test(max_episodes=10,e_greedy=0.9)
    # 绘制检测数据
    trainer.draw_step_plot()


def run_maze():
    env = Maze_Env()  # 创建环境
    # agent = Maze_Agent_Q_Learning() # 创建Agent
    agent = Maze_Agent_Sarsa()
    # trainer = Maze_Trainer_Q_Learning(env, agent)  # 创建训练器
    trainer = Maze_Trainer_Sarsa(env, agent)
    # 训练agent模型
    # env.after(100, trainer.train(max_episodes=10))  # 在窗口主循环中添加方法
    trainer.train(max_episodes=5)
    # 测试agent模型
    trainer.test(max_episodes=5,e_greedy=0.9)
    # 绘制检测数据
    trainer.draw_step_plot()
    env.mainloop()  # 调用主循环显示窗口


if __name__ == '__main__':
    #run_maze()
    run_line()
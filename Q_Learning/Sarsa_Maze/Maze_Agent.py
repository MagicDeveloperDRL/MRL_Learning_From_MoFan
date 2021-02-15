"""
@file: Maze_Agent.py
@author: MRL Liu
@time: 2021/2/15 15:43
@env: Python,Numpy
@desc:Maze项目的Agent
@ref:
@blog: https://blog.csdn.net/qq_41959920
"""
import numpy as np
import pandas as pd

class Maze_Agent(object):
    def __init__(self):
        self.actions = ['up','down','left','right']
        self.epsilon = 0.9  # 贪心系数
        self.gamma = 0.9  # 折扣因子
        self.learning_rate = 0.01  # 学习率
        # 初始化Q表
        self.q_table = self.build_q_table(self.actions)

    "策略函数，返回采取的动作"
    def choose_action(self, state):
        self.check_state_exist(state)
        # 有1-epsilon的概率随机选择动作
        if np.random.uniform() > self.epsilon:  # act non-greedy or state-action have no value
            action_name = np.random.choice(self.actions)
        # 有epsilon的概率贪心选取回报值最多的动作时
        else:
            state_actions = self.q_table.loc[state, :]  # 获取当前状态可采取的动作及其价值
            action_name = np.random.choice(state_actions[state_actions == np.max(state_actions)].index)
        return action_name

    "学习函数，值迭代"
    def learn(self, s, a, r, s_,a_):
        self.check_state_exist(s_)
        # 获取Q预测
        q_predict = self.q_table.loc[s, a]
        # 计算Q目标
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, a_]  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        # 更新Q表
        self.q_table.loc[s, a] += self.learning_rate * (q_target - q_predict)  # update

    "创建Q表"
    def build_q_table(self,actions):
        table = pd.DataFrame(
            data= None,  # q_table initial values
            index = None, # 行名为空
            columns = actions,  # 列名为actions's name
            dtype = np.float64
        )
        # print(table)    # show table
        return table

    "检查状态是否存在"
    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )
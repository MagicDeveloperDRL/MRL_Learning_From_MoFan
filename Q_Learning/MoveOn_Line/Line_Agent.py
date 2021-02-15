"""
@file: Agent_Brain.py
@author: MRL Liu
@time: 2021/2/13 19:12
@env: Python,Numpy
@desc:MoveOn_Line的Agent
@ref:
@blog: https://blog.csdn.net/qq_41959920
"""
import numpy as np
import pandas as pd

class Line_Agent(object):
    def __init__(self):
        self.n_states = 6
        self.actions = ['left', 'right']
        self.epsilon = 0.9  # 贪心系数
        self.gamma = 0.9  # 折扣因子
        self.learning_rate = 0.1  # 学习率
        # 初始化Q表
        self.q_table = self.build_q_table(self.n_states,self.actions)


    "策略函数，返回采取的动作"
    def choose_action(self,state):
        state_actions = self.q_table.iloc[state, :] # 获取当前状态可采取的动作及其价值
        # 有1-epsilon的概率或者当前状态没有对应动作时
        if (np.random.uniform() > self.epsilon) or (
        (state_actions == 0).all()):  # act non-greedy or state-action have no value
            action_name = np.random.choice(self.actions)
        # 有epsilon的概率贪心选取回报值最多的动作时
        else:
            action_name = state_actions.idxmax()
        return action_name

    "学习函数，值迭代"
    def learn(self,s, a, r, s_):
        # 获取Q预测
        q_predict = self.q_table.loc[s, a]
        # 计算Q目标
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        # 更新Q表
        self.q_table.loc[s, a] += self.learning_rate * (q_target - q_predict)  # update

    "创建Q表"
    def build_q_table(self,n_states, actions):
        table = pd.DataFrame(
            np.zeros((n_states, len(actions))),  # q_table initial values
            columns=actions,  # actions's name
        )
        # print(table)    # show table
        return table
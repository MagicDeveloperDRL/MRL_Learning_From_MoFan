"""
@file: Env.py
@author: MRL Liu
@time: 2021/2/13 19:12
@env: Python,Numpy
@desc:MoveOn_Line的环境
@ref:
@blog: https://blog.csdn.net/qq_41959920
"""
import time


class Line_Env(object):
    def __init__(self):
        self.actions = ['left', 'right'] # 动作空间
        self.n_states = 6 # 状态空间长度
        self.epsilon = 0.9 # 贪心系数
        self.fresh_time = 0.3 # 每次移动的更新时间

    "重置环境，返回初始状态"
    def reset(self):
        return 0

    "环境反馈，返回初始状态"
    def step(self,s,a):
        if a == 'right':  # 向右移动
            if s == self.n_states - 2:  # 达到目的地
                done = True
                s_ = s
                r = 1
            else:
                done = False
                s_ = s + 1
                r = 0
        else:  # move left
            done = False
            r = 0
            if s == 0:
                s_ = s  # 到达最左边
            else:
                s_ = s - 1
        return s_, r, done

    "环境更新"
    def render(self,S, episode, step_counter,done):
        # This is how environment be updated
        env_list = ['-'] * (self.n_states - 1) + ['T']  # '---------T' our environment
        if done == True:
            interaction = 'Episode %s: total_steps = %s' % (episode + 1, step_counter)
            print('\r{}'.format(interaction), end='')
            time.sleep(2)
            print('\r                                ', end='')
        else:
            env_list[S] = 'o'
            interaction = ''.join(env_list)
            print('\r{}'.format(interaction), end='')
            time.sleep(self.fresh_time)









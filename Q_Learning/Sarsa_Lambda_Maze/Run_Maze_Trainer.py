"""
@file: Maze_Trainer.py
@author: MRL Liu
@time: 2021/2/15 15:43
@env: Python,Numpy
@desc:Maze项目的训练器
@ref:
@blog: https://blog.csdn.net/qq_41959920
"""
from Q_Learning.Sarsa_Lambda_Maze import Maze_Env
from Q_Learning.Sarsa_Lambda_Maze import Maze_Agent

from matplotlib import pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #使用中文字符
plt.rcParams['axes.unicode_minus'] = False #显示负数的负号

class Maze_Trainer(object):
    def __init__(self,env,agent):
        self.env = env
        self.agent = agent
        # 每回合最大步数记录器
        self.list_n_step_train = []
        self.list_n_step_test = []

    def train(self,max_episodes,is_Learn = True):
        if is_Learn == True:
            print('\n仿真训练启动...')
        for episode in range(max_episodes):
            step_counter = 0

            # 获取初始环境状态
            observation = self.env.reset()
            action = self.agent.choose_action(str(observation))  # agent根据当前状态采取动作
            self.agent.eligibility_trace *= 0
            # 开始本回合的仿真
            while True:
                self.env.render()
                # 获取动作和环境反馈
                observation_, reward, done = self.env.step(observation, action)  # env根据动作做出反馈
                # 学习本回合的经验(s, a, r, s, a) ==> Sarsa
                action_ = self.agent.choose_action(str(observation_))  # agent根据当前状态采取动作
                if is_Learn:
                    self.agent.learn(str(observation), action_, reward, str(observation_),action_)
                # 当前状态发生切换
                observation = observation_
                action = action_
                # 更新环境
                step_counter += 1


                # 检测本回合是否需要停止
                if done:
                    if is_Learn:
                        self.list_n_step_train.append(step_counter)  # 记录最大回合数
                    else:
                        self.list_n_step_test.append(step_counter)  # 记录最大回合数
                    print('\n第{}回合仿真结束,共尝试了{}步'.format(episode+1,step_counter))
                    break
        if is_Learn == True:
            print('\n仿真训练结束')

    def test(self,max_episodes=1):
        print('\n仿真测试启动...')
        self.agent.epsilon = 0.9  # 设置贪心指数为最大
        self.train(max_episodes=max_episodes, is_Learn=False)
        print('\n仿真测试结束')

    def draw_step_plot(self):
        # 创建画布
        fig = plt.figure(figsize=(6, 6))  # 创建一个指定大小的画布
        # 创建画布
        print('绘制数据')
        # 添加第1个窗口
        ax1 = fig.add_subplot(111)  # 添加一个1行1列的序号为1的窗口
        # 添加标注
        ax1.set_title('训练中的最大步数变化状况', fontsize=14)  # 设置标题
        ax1.set_xlabel('x-回合数', fontsize=14, fontfamily='sans-serif', fontstyle='italic')
        ax1.set_ylabel('y-最大步数', fontsize=14, fontstyle='oblique')
        # 绘制函数
        x_data_train = range(1,len(self.list_n_step_train)+1)
        y_data_train = self.list_n_step_train
        x_data_test = range(1,len(self.list_n_step_test)+1)
        y_data_test = self.list_n_step_test
        line1, = ax1.plot(x_data_train, y_data_train, color='blue', label="训练值")
        line2, = ax1.plot(x_data_test, y_data_test, color='red', label="应用值")
        ax1.legend(handles=[line1, line2], loc=1)  # 绘制图例说明
        plt.grid(True)  # 启用表格

if __name__ == '__main__':
    env = Maze_Env() # 创建环境
    agent = Maze_Agent() # 创建Agent
    trainer = Maze_Trainer(env, agent)  # 创建训练器
    # 训练agent模型
    #env.after(100, trainer.train(max_episodes=10))  # 在窗口主循环中添加方法
    trainer.train(max_episodes=30)
    # 测试agent模型
    trainer.test(max_episodes=30)
    # 绘制检测数据
    trainer.draw_step_plot()
    env.mainloop()  # 调用主循环显示窗口







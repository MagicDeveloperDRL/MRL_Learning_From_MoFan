"""
@file: Run_Trainer.py
@author: MRL Liu
@time: 2021/2/13 19:13
@env: Python,Numpy
@desc:MoveOn_Line的训练器
@ref:
@blog: https://blog.csdn.net/qq_41959920
"""


from Q_Learning.MoveOn_Line import Line_Env
from Q_Learning.MoveOn_Line import Line_Agent
from matplotlib import pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #使用中文字符
plt.rcParams['axes.unicode_minus'] = False #显示负数的负号

class Line_Trainer(object):
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
            # 获取初始环境状态
            observation = self.env.reset()
            step_counter = 0
            done =False
            self.env.render(observation, episode, step_counter,done)
            # 开始本回合的仿真
            while True:
                # 获取动作和环境反馈
                action = self.agent.choose_action(observation)# agent根据当前状态采取动作
                observation_, reward, done = self.env.step(observation,action)# env根据动作做出反馈
                # 学习本回合的经验
                if is_Learn:
                    self.agent.learn(observation, action, reward, observation_)
                # 当前状态发生切换
                observation = observation_
                # 更新环境
                step_counter += 1
                self.env.render(observation, episode, step_counter, done)  # env更新环境

                # 检测本回合是否需要停止
                if done:
                    if is_Learn:
                        self.list_n_step_train.append(step_counter) # 记录最大回合数
                    else:
                        self.list_n_step_test.append(step_counter) # 记录最大回合数
                    break
        if is_Learn == True:
            print('\n仿真训练结束')

    def test(self,max_episodes=1):
        print('\n仿真测试启动...')
        self.agent.epsilon = 0.9 # 设置贪心指数为最大
        self.train(max_episodes=max_episodes,is_Learn=False)
        print('\n仿真测试结束')

    def draw_step_plot(self):
        # 创建画布
        fig = plt.figure(figsize=(12, 12))  # 创建一个指定大小的画布

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
        ax1.legend(handles=[line1, line2], loc=4)  # 绘制图例说明
        plt.grid(True)  # 启用表格


if __name__ == "__main__":
    env = Line_Env() # 创建环境
    agent = Line_Agent() # 创建agent
    trainer  = Line_Trainer(env,agent)# 创建训练器
    # 训练agent模型
    trainer.train(max_episodes=50)
    # 使用agent模型
    trainer.test(max_episodes=50)
    # 绘制检测数据
    trainer.draw_step_plot()






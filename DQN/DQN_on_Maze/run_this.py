from DQN.DQN_on_Maze.maze_env import Maze
from DQN.DQN_on_Maze.Agent_DQN import Agent_DQN


# 在Maze环境下运行DQN算法的训练算法
def run_DQN_on_Maze():
    step = 0
    # 进行100次训练
    for episode in range(300):
        observation = env.reset()
        while(True):
            # 刷新环境
            env.render()
            # 基于观测值选择动作
            action = agent.choose_Action(observation)
            # 获取环境的反馈
            observation_,r,done = env.step(action)
            # 将转换元组存入记忆池
            agent.store_in_memory(observation,action,r,observation_)
            # 以固定频率进行学习
            if(step>200) and (step %5 ==0):
                agent.learn_from_step()
            # 切换观察值
            observation =observation_
            # 如果回合运行结束：
            if done:
                print('第{0}次回合结束'.format(episode + 1))
                break
            step += 1

    # 训练结束
    print("DQN算法测试程序训练结束")

if __name__=="__main__":
    # 创建环境模块
    env = Maze()
    # 创建Agent模块
    agent = Agent_DQN(env.n_features,env.n_actions)
    # 启动DQN的训练算法
    env.after(100,run_DQN_on_Maze())
    env.mainloop()
    agent.plot_cost()

import gym
from MRL_Learning_From_MoFan.DQN_on_Maze.Agent_DQN import Agent_DQN


# 在Gym环境下运行DQN算法的训练算法
def run_DQN_on_Gym():
    total_steps = 0  # 记录步数
    # 进行100回合的训练
    for i_episode in range(100):
        observation = env.reset()
        ep_r = 0 #该回合的奖励值
        #开始步数
        while True:
            # 刷新环境
            env.render()
            # 基于观测值选择动作
            action = agent.choose_Action(observation)
            # 获取环境的反馈
            observation_, reward, done, info = env.step(action)
            x, x_dot, theta, theta_dot = observation_
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            reward = r1 + r2
            # 将转换元组存入记忆池
            agent.store_in_memory(observation, action, reward, observation_)
            # 以固定频率进行学习
            if total_steps > 1000:
                agent.learn_from_step()
            # 切换观察值
            observation = observation_
            ep_r += reward
            # 如果回合运行结束：
            if done:
                print('Episode:', i_episode,
                      'ep_r', round(ep_r, 2),
                      'epsilon:', round(agent.epsilon, 2))
                break
            total_steps += 1
    # 训练结束
    print("DQN算法测试程序训练结束")


if __name__=="__main__":
    # 导入Gym库中的一个环境
    env = gym.make('CartPole-v0')
    env = env.unwrapped  # 不做这个会有很多限制

    # print(env.action_space)
    # print(env.observation_space)
    # print(env.observation_space.high)
    # print(env.observation_space.low)
    agent = Agent_DQN(
        n_actions=env.action_space.n,
        n_features=env.observation_space.shape[0],
        learning_rate=0.01,
        e_greedy=0.9,
        replace_target_iter=100,
        memory_size=2000,
        e_greedy_increment=0.0008
    )
    # 开始训练算法
    run_DQN_on_Gym()
    # 最后输出cost曲线
    agent.plot_cost()
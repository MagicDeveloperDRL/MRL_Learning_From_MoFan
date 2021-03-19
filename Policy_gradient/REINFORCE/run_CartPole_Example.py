'''''''''
@file: run_CartPole_Example.py
@author: MRL Liu
@time: 2021/3/12 10:56
@env: Python,Numpy
@desc:
@ref:
@blog: https://blog.csdn.net/qq_41959920
'''''''''
import gym
import matplotlib.pyplot as plt
from Policy_gradient.REINFORCE.Agent_REINFORCE import Agent_REINFORCE

DISPLAY_REWARD_THRESHOLD = 400  # renders environment if total episode reward is greater then this threshold
RENDER = False  # 是否渲染环境，渲染环境会使训练过程变慢
TEST_ENV_NAME ='CartPole-v0'

def train(agent,max_episodes=3000,max_steps_each_episode=1000,is_Render = False):
    for i_episode in range(max_episodes):
        running_reward = 0 # 运行的回合奖励
        t_step = 0
        observation = env.reset() # 初始化环境状态

        while True:
            if is_Render: env.render() # 是否渲染环境
            action = agent.choose_action(observation) # 选择动作
            observation_, reward, done, info = env.step(action) # 获取反馈
            agent.store_in_memory(observation, action, reward) # 存储
            observation = observation_ # 切换观察值
            t_step +=1
            if done or t_step>=max_steps_each_episode:
                ep_rs_sum = sum(agent.ep_rs)#获取本回合的累计奖励
                vt = agent.learn()  # 该训练回合结束时才开始学习
                # 打印信息
                if 'running_reward' not in globals():
                    running_reward = ep_rs_sum
                else:
                    running_reward = running_reward * 0.99 + ep_rs_sum * 0.01
                #if running_reward > DISPLAY_REWARD_THRESHOLD: is_Render = True  # rendering
                print("episode:", i_episode, "  reward:", int(running_reward))


                if i_episode == 0:
                    plt.plot(vt)  # plot the episode vt
                    plt.xlabel('episode steps')
                    plt.ylabel('normalized state-action value') # 长期回报的值
                    plt.show()
                break




if __name__=='__main__':
    env = gym.make(TEST_ENV_NAME)
    env = env.unwrapped
    env.seed(1)

    print(env.action_space)
    print(env.observation_space)
    print(env.observation_space.high)
    print(env.observation_space.low)

    STATE_FEATURES = env.observation_space.shape[0]
    ACTION_SPACE = env.action_space.n
    print("ACTION_SPACE:",ACTION_SPACE)

    agent = Agent_REINFORCE(
        n_features = STATE_FEATURES,
        n_actions = ACTION_SPACE,
        learning_rate=0.02,
        reward_decay=0.99,
        output_graph=False,
    )

    train(agent, num_episode=250)
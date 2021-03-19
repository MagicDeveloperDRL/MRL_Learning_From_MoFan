'''''''''
@file: Agent_REINFORCE.py
@author: MRL Liu
@time: 2021/3/12 10:47
@env: Python,Numpy
@desc:
@ref: 交叉熵损失：https://blog.csdn.net/b1055077005/article/details/100152102
@blog: https://blog.csdn.net/qq_41959920
'''''''''
import numpy as np
import tensorflow as tf


np.random.seed(1)
tf.set_random_seed(1)


class Agent_REINFORCE(object):
    def __init__(self,
                    n_features,
                    n_actions,
                    learning_rate=0.01,
                    reward_decay=0.95,
                    output_graph=False,
                    sess = None
                 ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.gamma = reward_decay
        # 初始化信息列表
        self.ep_obs = [] # 存储每回合的状态
        self.ep_as  = [] # 存储每回合的动作
        self.ep_rs =  [] # 存储每回合的奖励
        # 初始化网络
        self._build_net()
        # 初始化会话
        if sess is None:
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
        else:
            self.sess = sess
        if output_graph:
            # $ tensorboard --logdir=logs
            # http://0.0.0.0:6006/
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)
        # 存储历史检测数据
        self.cost_his = []  # 存储历史损失

    def _build_net(self):
        with tf.name_scope('inputs'):
            self.tf_obs = tf.placeholder(tf.float32, [None, self.n_features], name="observations") # 网络输入
            self.tf_acts = tf.placeholder(tf.int32, [None, ], name="actions") # 动作值
            self.tf_vt = tf.placeholder(tf.float32, [None, ], name="reward") # 状态值
        # fc1 生成一个全连接层
        layer = tf.layers.dense(
            inputs=self.tf_obs, # 输入数据
            units=10, # 神经元数量
            activation=tf.nn.tanh,  # tanh 激活函数
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),# 权重矩阵的初始化器
            bias_initializer=tf.constant_initializer(0.1),# 偏置项的初始化器
            name='fc1'
        )
        # fc2
        all_act = tf.layers.dense(
            inputs=layer,
            units=self.n_actions,
            activation=None,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc2'
        )

        self.all_act_prob = tf.nn.softmax(all_act, name='act_prob')  # use softmax to convert to probability
        #print('self.all_act_prob.shape:',self.all_act_prob.shape) # shape=(?,2)
        #print('self.all_act_prob:', self.all_act_prob)
        with tf.name_scope('loss'):
            # to maximize total reward (log_p * R) is to minimize -(log_p * R), and the tf only have minimize(loss)
            # 计算logits和labels之间的稀疏softmax交叉熵。
            #neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=all_act,
                                                                          #labels=self.tf_acts)  # this is negative log of chosen action
            # or in this way:
            neg_log_prob = tf.reduce_sum(-tf.log(self.all_act_prob)*tf.one_hot(self.tf_acts, self.n_actions), axis=1)# 求和，shape=[?,3]变成了shape=[?,1]
            loss = tf.reduce_mean(neg_log_prob * self.tf_vt)  # 乘以一个状态价值，相当于区分不同时刻的重要程度，求平均值，shape=[?,1]变成了shape=[1]reward guided loss

        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

    def choose_action(self, observation):
        prob_weights = self.sess.run(self.all_act_prob, feed_dict={self.tf_obs: observation[np.newaxis, :]})
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())  # select action w.r.t the actions prob
        print('shape[1]:', prob_weights.shape[1]) # 2
        print('ravel:', prob_weights.ravel())# 概率
        print('action:',action)
        return action

    def store_in_memory(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def __clear_memory(self):
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []

    def learn(self):
        # 获取折扣后的长期回报  discount and normalize episode reward
        discounted_ep_rs_norm = self._discount_and_norm_rewards()

        # 训练策略网络
        self.sess.run(self.train_op, feed_dict={
             self.tf_obs: np.vstack(self.ep_obs), # 给计算图输入观察值，shape=[None, n_obs]
             self.tf_acts: np.array(self.ep_as),  # 给计算图输入动作，shape=[None, ]
             self.tf_vt: discounted_ep_rs_norm,  # 状态价值，给计算图输入折扣回报，shape=[None, ]
        })

        self.__clear_memory() # 清空本回合的数据

        return discounted_ep_rs_norm # 返回归一化的长期回报数据

    def _discount_and_norm_rewards(self):
        # 折扣的回合奖励（使用了一个技巧），discount episode rewards
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add

        # 将长期回报全部均值方差归一化，为了使训练效果更好
        discounted_ep_rs -= np.mean(discounted_ep_rs) # 求出平均值
        discounted_ep_rs /= np.std(discounted_ep_rs) # 除以标准差

        return discounted_ep_rs

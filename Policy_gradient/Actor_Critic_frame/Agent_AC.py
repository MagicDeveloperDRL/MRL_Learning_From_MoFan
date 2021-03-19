'''''''''
@file: Agent_AC.py
@author: MRL Liu
@time: 2021/3/17 15:16
@env: Python,Numpy
@desc:
@ref:
@blog: https://blog.csdn.net/qq_41959920
'''''''''
import numpy as np
import tensorflow as tf


class Actor(object):
    def __init__(self, sess, n_features, n_actions, lr=0.001):
        self.sess = sess
        self.n_features = n_features
        self.n_actions = n_actions
        self.learning_rate = lr
        # 初始化策略网络
        self.__init_net()


    def __init_net(self):
        self.s = tf.placeholder(tf.float32, [1, self.n_features], "state")
        self.a = tf.placeholder(tf.int32, None, "act")
        self.td_error = tf.placeholder(tf.float32, None, "td_error")  # TD_error

        with tf.variable_scope('Actor'):
            # 第1个全连接层
            l1 = tf.layers.dense(
                inputs=self.s,
                units=20,  # number of hidden units
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )
            # 第2个全连接层
            self.acts_prob = tf.layers.dense(
                inputs=l1,
                units=self.n_actions,  # output units
                activation=tf.nn.softmax,  # get action probabilities
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='acts_prob'
            )

        with tf.variable_scope('exp_v'):
            log_prob = tf.log(self.acts_prob[0, self.a])
            self.exp_v = tf.reduce_mean(log_prob * self.td_error)  # advantage (TD_error) guided loss

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(-self.exp_v)  # minimize(-exp_v) = maximize(exp_v)

    def learn(self, s, a, td):
        s = s[np.newaxis, :] # 升维，从一维变为二维
        # 训练网络
        feed_dict = {self.s: s, self.a: a, self.td_error: td}
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)
        return exp_v

    def choose_action(self, s):
        s = s[np.newaxis, :] # 升维，从一维变为二维
        probs = self.sess.run(self.acts_prob, {self.s: s})   # 运行网络，获取动作概率shape=(1,n_actions)
        action = np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())
        return action


class Critic(object):
    def __init__(self, sess, n_features, lr=0.01,gamma=0.9):
        self.sess = sess
        self.n_features = n_features
        self.learning_rate = lr
        self.gamma = gamma
        # 初始化网络
        self.__init_net()

    def __init_net(self):
        self.s = tf.placeholder(tf.float32, [1, self.n_features], "state")
        self.v_ = tf.placeholder(tf.float32, [1, 1], "v_next") # 真实价值
        self.r = tf.placeholder(tf.float32, None, 'r')

        with tf.variable_scope('Critic'):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=20,  # number of hidden units
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )
            # 预测价值
            self.v = tf.layers.dense(
                inputs=l1,
                units=1,  # output units
                activation=None,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='V'
            )

        with tf.variable_scope('squared_TD_error'):
            self.td_error = self.r + self.gamma * self.v_ - self.v
            self.loss = tf.square(self.td_error)  # TD_error = (r+gamma*V_next) - V_eval
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def learn(self, s, r, s_):
        s, s_ = s[np.newaxis, :], s_[np.newaxis, :]

        v_ = self.sess.run(self.v, {self.s: s_}) # 用自身策略网络来估计v_
        td_error, _ = self.sess.run([self.td_error, self.train_op],
                                          {self.s: s, self.v_: v_, self.r: r})
        return td_error

# 处理连续动作的Actor
class Actor_Continue(object):
    def __init__(self, sess, n_features, action_bound, lr=0.0001):
        self.sess = sess
        self.n_features = n_features
        self.action_bound = action_bound
        self.learning_rate = lr
        # 初始化策略网络
        self.__init_net()


    def __init_net(self):
        self.s = tf.placeholder(tf.float32, [1, self.n_features], "state")
        self.a = tf.placeholder(tf.float32, None, name="act")
        self.td_error = tf.placeholder(tf.float32, None, name="td_error")  # TD_error

        with tf.variable_scope('Actor'):
            # 第1个全连接层
            l1 = tf.layers.dense(
                inputs=self.s,
                units=30,  # number of hidden units
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )
            # 第2个全连接层
            mu = tf.layers.dense(
                inputs=l1,
                units=1,  # number of hidden units
                activation=tf.nn.tanh,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='mu'
            )
            # 第2个全连接层
            sigma = tf.layers.dense(
                inputs=l1,
                units=1,  # output units
                activation=tf.nn.softplus,  # get action probabilities
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(1.),  # biases
                name='sigma'
            )
        # 下面定义该网络最终的输出动作
        global_step = tf.Variable(0, trainable=False)
        # self.e = epsilon = tf.train.exponential_decay(2., global_step, 1000, 0.9)
        #print("mu:{},sigma{}".format(mu, sigma))
        self.mu, self.sigma = tf.squeeze(mu * 2), tf.squeeze(sigma + 0.1)# 降低维度,最后变成0维，即单纯的数值
        #print("mu:{},sigma{}".format(self.mu,self.sigma))
        self.normal_dist = tf.distributions.Normal(self.mu, self.sigma)
        print("sample:{}".format(self.normal_dist.sample(1)))
        self.action = tf.clip_by_value(self.normal_dist.sample(1), self.action_bound[0], self.action_bound[1])

        with tf.name_scope('exp_v'):
            log_prob = self.normal_dist.log_prob(self.a)  # loss without advantage
            self.exp_v = log_prob * self.td_error  # advantage (TD_error) guided loss
            # Add cross entropy cost to encourage exploration
            self.exp_v += 0.01 * self.normal_dist.entropy()

        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(-self.exp_v, global_step)# min(v) = max(-v)

    def learn(self, s, a, td):
        s = s[np.newaxis, :]
        feed_dict = {self.s: s, self.a: a, self.td_error: td}
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)
        return exp_v

    def choose_action(self, s):
        s = s[np.newaxis, :]
        action = self.sess.run(self.action, {self.s: s})
        print('action:',action)
        return  action # get probabilities for all actions

# 处理连续动作的Critic
class Critic_Continue(object):
    def __init__(self, sess, n_features, lr=0.01,gamma=0.9):
        self.sess = sess
        self.n_features = n_features
        self.learning_rate = lr
        self.gamma = gamma
        # 初始化网络
        self.__init_net()

    def __init_net(self):
        with tf.name_scope('inputs'):
            self.s = tf.placeholder(tf.float32, [1, self.n_features], "state")
            self.v_ = tf.placeholder(tf.float32, [1, 1], name="v_next")
            self.r = tf.placeholder(tf.float32,None, name='r')

        with tf.variable_scope('Critic'):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=30,  # number of hidden units
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )
            # 预测价值
            self.v = tf.layers.dense(
                inputs=l1,
                units=1,  # output units
                activation=None,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='V'
            )

        with tf.variable_scope('squared_TD_error'):
            self.td_error = tf.reduce_mean(self.r + self.gamma * self.v_ - self.v)
            self.loss = tf.square(self.td_error)  # TD_error = (r+gamma*V_next) - V_eval
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def learn(self, s, r, s_):
        s, s_ = s[np.newaxis, :], s_[np.newaxis, :]

        v_ = self.sess.run(self.v, {self.s: s_}) # 用自身策略网络来估计v_
        td_error, _ = self.sess.run([self.td_error, self.train_op],
                                    {self.s: s, self.v_: v_, self.r: r})
        return td_error

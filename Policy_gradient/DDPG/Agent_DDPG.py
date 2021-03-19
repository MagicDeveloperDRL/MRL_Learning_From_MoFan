'''''''''
@file: Agent_DDPG.py
@author: MRL Liu
@time: 2021/3/18 19:45
@env: Python,Numpy
@desc:使用DDPG（深度确定性策略梯度算法）作为学习算法的Agent
@ref:
@blog: https://blog.csdn.net/qq_41959920
'''''''''
import tensorflow as tf
import numpy as np


class Agent_DDPG(object):
    def __init__(self, state_dim,action_dim, action_bound,memory_size,batch_size,a_lr,c_lr, gamma,replacement,sess):
        self.a_dim, self.s_dim, self.a_bound = action_dim, state_dim, action_bound,
        self.a_lr = a_lr
        self.c_lr = c_lr
        self.batch_size = batch_size
        self.gamma = gamma
        self.replacement = replacement
        if sess is None:
            self.sess = tf.Session()
        else:
            self.sess = sess
        # 创建经验池
        self.memory_size = memory_size
        self.memory = Memory(capacity=memory_size,dims= state_dim * 2 + action_dim + 1)
        # 定义网络输入
        self.S = tf.placeholder(tf.float32, [None, self.s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, self.s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')
        # 初始化Actor网络
        with tf.variable_scope('Actor'):
            self.a = self._get_actor(self.S,scope='eval_net',trainable=True)
            self.a_ = self._get_actor(self.S_, scope='target_net', trainable=False)
        # 初始化Critic网络
        with tf.variable_scope('Critic'):
            self.q = self._get_critic(self.S, self.a, 'eval_net', trainable=True)
            # Input (s_, a_), output q_ for q_target
            self.q_ = self._get_critic(self.S_, self.a_, 'target_net', trainable=False)
        # 网络参数
        self.a_e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval_net')
        self.a_t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target_net')
        self.c_e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval_net')
        self.c_t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target_net')
        if self.replacement['name'] == 'hard':
            self.a_t_replace_counter = 0
            self.a_hard_replace = [tf.assign(t, e) for t, e in zip(self.a_t_params, self.a_e_params)]
        else:
            self.soft_replace = [tf.assign(t, (1 - self.replacement['tau']) * t + self.replacement['tau'] * e)
                                 for t, e in zip(self.a_t_params + self.c_t_params, self.a_e_params+ self.c_e_params)]


        with tf.variable_scope('target_q'):
            self.target_q = self.R + self.gamma * self.q_

        with tf.variable_scope('TD_error'):
            self.td_error = tf.reduce_mean(tf.squared_difference(self.target_q, self.q))

        with tf.variable_scope('C_train'):
            self.c_train_op = tf.train.AdamOptimizer(self.c_lr).minimize(self.td_error, var_list=self.c_e_params)

        # 定义actor
        with tf.variable_scope('a_loss'):
            a_loss = - tf.reduce_mean(self.q)    # maximize the q

        with tf.variable_scope('A_train'):
            self.a_train_op = tf.train.AdamOptimizer(self.a_lr).minimize(a_loss, var_list=self.a_e_params)
        self.sess.run(tf.global_variables_initializer())

    def _get_actor(self,state,scope,trainable):
        with tf.variable_scope(scope):
            init_w = tf.random_normal_initializer(0., 0.3)
            init_b = tf.constant_initializer(0.1)
            layer_1 = tf.layers.dense(inputs = state,
                                      units = 30,
                                      activation= tf.nn.relu,
                                      kernel_initializer=init_w,
                                      bias_initializer=init_b,
                                      name='layer_1',
                                      trainable = trainable,
                                      )
            with tf.variable_scope('a'):
                actions = tf.layers.dense(inputs =layer_1,
                                             units=self.a_dim,
                                             activation=tf.nn.tanh,
                                             kernel_initializer=init_w,
                                             bias_initializer=init_b,
                                             name='a',
                                             trainable=trainable)
                scaled_a = tf.multiply(actions, self.a_bound, name='scaled_a')
        return scaled_a

    def _get_critic(self,state,action,scope,trainable):
        with tf.variable_scope(scope):
            init_w = tf.random_normal_initializer(0., 0.1)
            init_b = tf.constant_initializer(0.1)
            with tf.variable_scope('layer_1'):
                n_l1 = 30
                w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], initializer=init_w, trainable=trainable)
                w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], initializer=init_w, trainable=trainable)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=init_b, trainable=trainable)
                layer_1 = tf.nn.relu(tf.matmul(state, w1_s) + tf.matmul(action, w1_a) + b1)

            with tf.variable_scope('q'):
                q = tf.layers.dense(inputs=layer_1,
                                    units=1,
                                    kernel_initializer=init_w,
                                    bias_initializer=init_b,
                                    trainable=trainable)  # Q(s,a)
        return q

    def store_transition(self, s, a, r, s_):
        self.memory.store_transition(s, a, r, s_)

    def learn(self):
        batch_data = self.memory.sample(self.batch_size)
        batch_s = batch_data [:,:self.s_dim]
        batch_a = batch_data [:,self.s_dim:self.s_dim+self.a_dim]
        batch_r = batch_data[:, -self.s_dim-1:-self.s_dim]
        batch_s_ = batch_data[:, -self.s_dim:]
        self.sess.run(self.a_train_op,{self.S:batch_s})
        self.sess.run(self.c_train_op,{self.S:batch_s,self.a:batch_a,self.R:batch_r,self.S_:batch_s_})

        # soft target replacement
        self.sess.run(self.soft_replace)

    def choose_action(self,s):
        s = s[np.newaxis, :]  # single state
        return self.sess.run(self.a,feed_dict={self.S:s})[0]

# 记忆池
class Memory(object):
    def __init__(self, capacity, dims):
        self.capacity = capacity # 容量大小
        self.data = np.zeros((capacity, dims),dtype=np.float32) # 数据
        self.pointer = 0 # 当前指针

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_)) # 按行连接
        index = self.pointer % self.capacity  # replace the old memory with new memory
        self.data[index, :] = transition
        self.pointer += 1

    def sample(self, n):
        assert self.pointer >= self.capacity, 'Memory has not been fulfilled'
        indices = np.random.choice(self.capacity, size=n) # 从记忆池中随机采样n个数
        return self.data[indices, :] # 获取n个采样
'''''''''
@file: Agent_Dueling_DQN.py
@author: MRL Liu
@time: 2021/3/10 13:13
@env: Python,Numpy
@desc:
@ref:
@blog: https://blog.csdn.net/qq_41959920
'''''''''
import numpy as np
import tensorflow as tf

np.random.seed(1)
tf.set_random_seed(1)

class Agent_Dueling_DQN(object):
    def __init__(self,
                 n_actions,
                 n_features,
                 learning_rate=0.001,
                 reward_decay=0.9,
                 e_greedy=0.9,
                 replace_target_iter=200,
                 memory_size=500,
                 batch_size=32,
                 e_greedy_increment=None,
                 output_graph=False,
                 sess=None,
                 ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.learn_rate = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        # 初始化记忆池
        self.init_memory()
        # 初始化网络
        self.init_eval_net()
        self.init_target_net()
        # 定时复制参数给target_net
        self.learn_step_counter = 0
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]
        # 初始化会话
        if sess is None:
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
        else:
            self.sess = sess
        # 是否输出图
        if output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)
        # 存储损失
        self.cost_his = []  # 存储历史损失
        self.q_his = []  # 记录agent的动作价值
        self.running_q = 0

    def choose_action(self, s):
        # 将一维数组转换为二维数组，虽然只有一行
        s = s[np.newaxis, :]
        # 获取评估的动作价值
        action_values = self.sess.run(fetches=self.q_eval, feed_dict={self.s: s})

        # 记录agent的动作价值，便于观测
        if not hasattr(self, 'q_his'):
            self.q_his = []
            self.running_q = 0
        self.running_q = self.running_q * 0.99 + 0.01 * np.max(action_values)
        self.q_his.append(self.running_q)

        # greedy策略
        if np.random.uniform() > self.epsilon:  # 随机选择一个动作
            action = np.random.randint(0, self.n_actions)
        else:  # 选择最好动作
            action = np.argmax(action_values)
        return action

    # 初始化记忆池
    def init_memory(self):
        self.memory = np.zeros((self.memory_size, self.n_features * 2 + 2))
    # 向记忆池存入数据
    def store_in_memory(self,s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    # 从记忆池中取出一定数量的数据
    def pick_from_memory(self, batch_size):
        if self.memory_size < self.memory_counter:
            batch_indexs = np.random.choice(self.memory_size, size=batch_size)
        else:
            batch_indexs = np.random.choice(self.memory_counter, size=batch_size)
        batch_data = self.memory[batch_indexs, :]
        return batch_data

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('\ntarget_net的参数被更新\n')

        # 获取采样数据
        batch_memory =self.pick_from_memory(batch_size=self.batch_size)

        q_next = self.sess.run(self.q_next, feed_dict={self.s_: batch_memory[:, -self.n_features:]}) # next observation
        q_eval = self.sess.run(self.q_eval, feed_dict={self.s: batch_memory[:, :self.n_features]})

        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)

        reward = batch_memory[:, self.n_features + 1]

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)
        # 进行优化
        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.n_features],
                                                self.q_target: q_target})
        self.cost_his.append(self.cost)

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1
        return
    # 初始化网络
    def init_eval_net(self):
        # 定义网络的输入输出
        self.s = tf.placeholder(dtype=tf.float32,shape= [None,self.n_features],name='s')
        self.q_target = tf.placeholder(dtype=tf.float32,shape=[None,self.n_actions],name='Q_target')
        with tf.variable_scope('eval_net'):
            # 定义神经层的配置
            n_Layer1=20
            c_names=['eval_net_params',tf.GraphKeys.GLOBAL_VARIABLES]
            w_initializer = tf.random_normal_initializer(0.,0.3)
            b_initializer = tf.constant_initializer(0.1)
            self.q_eval = self.__create_fc_layer(self.s,n_Layer1,w_initializer,b_initializer,c_names)
            #搭建损失函数和优化器
            with tf.variable_scope('loss'):
                self.loss = tf.reduce_mean(tf.squared_difference(self.q_target,self.q_eval))
            with tf.variable_scope('train'):
                self._train_op = tf.train.RMSPropOptimizer(self.learn_rate).minimize(self.loss)
    def init_target_net(self):
        # 定义网络的输入输出
        self.s_ = tf.placeholder(dtype=tf.float32,shape= [None,self.n_features],name='s_')
        with tf.variable_scope('target_net'):
            # 定义神经层的配置
            n_Layer1=20
            c_names=['target_net_params',tf.GraphKeys.GLOBAL_VARIABLES]
            w_initializer = tf.random_normal_initializer(0.,0.3)
            b_initializer = tf.constant_initializer(0.1)

            self.q_next = self.__create_fc_layer(self.s_,n_Layer1,w_initializer,b_initializer,c_names)
    def __create_fc_layer(self,input,n_Layer,w_initializer,b_initializer,c_names):
        # 定义L1
        with tf.variable_scope('layer_1'):
            w1 = tf.get_variable(name='w1',
                                 shape=[self.n_features, n_Layer],
                                 initializer=w_initializer,
                                 collections=c_names)
            b1 = tf.get_variable(name='b1',
                                 shape=[1, n_Layer],
                                 initializer=b_initializer,
                                 collections=c_names)
            layer_1 = tf.nn.relu(tf.matmul(input, w1) + b1)

        # Dueling DQN
        with tf.variable_scope('Value'):
            w2 = tf.get_variable(name='w2',
                                 shape=[n_Layer, 1],
                                 initializer=w_initializer,
                                 collections=c_names)
            b2 = tf.get_variable(name='b2',
                                 shape=[1, 1],
                                 initializer=b_initializer,
                                 collections=c_names)
            self.V = tf.matmul(layer_1, w2) + b2 #shape=(?,1)

        with tf.variable_scope('Advantage'):
            w2 = tf.get_variable(name='w2',
                                 shape=[n_Layer, self.n_actions],
                                 initializer=w_initializer,
                                 collections=c_names)
            b2 = tf.get_variable(name='b2',
                                 shape=[1, self.n_actions],
                                 initializer=b_initializer,
                                 collections=c_names)
            self.A = tf.matmul(layer_1, w2) + b2 #shape=(?,self.n_actions)

        with tf.variable_scope('Q'):
            # out.shape=(?,self.n_actions),虽然self.A和self.V的维度不相同，但是这里会触发numpy的一个广播机制来进行计算
            out = self.V + (self.A - tf.reduce_mean(self.A, axis=1, keep_dims=True))  # Q = V(s) + A(s,a)

        return out

    # 显示图
    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()

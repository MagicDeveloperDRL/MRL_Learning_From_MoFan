'''''''''
@file: Agent_Prioritized_Replay_DQN.py
@author: MRL Liu
@time: 2021/3/10 14:51
@env: Python,Numpy
@desc:
@ref:
@blog: https://blog.csdn.net/qq_41959920
'''''''''
import numpy as np
import tensorflow as tf

np.random.seed(1)
tf.set_random_seed(1)

class SumTree(object):
    """
    求和数（二叉树类型），叶子节点存储优先级
    This SumTree code is a modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/SumTree.py

    Story data with its priority in the tree.
    """
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity  # for all priority values
        self.tree = np.zeros(2 * capacity - 1)
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        self.data = np.zeros(capacity, dtype=object)  # for all transitions
        # [--------------data frame-------------]
        #             size: capacity

    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data  # update data_frame
        self.update(tree_idx, p)  # update tree_frame

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity
            self.data_pointer = 0

    def update(self, tree_idx, p):
        # 为叶节点添加数值，再用叶节点更新父节点等
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # then propagate the change through tree
        while tree_idx != 0:    # this method is faster than the recursive loop in the reference code
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        """
        Tree structure and array storage:

        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions

        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_idx = 0
        while True:     # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx + 1         # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):        # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:       # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0]  # the root

class Memory(object):  # stored as ( s, a, r, s_ ) in SumTree
    """
    This Memory class is modified based on the original code from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """
    epsilon = 0.01  # small amount to avoid zero priority
    alpha = 0.6  # [0~1] convert the importance of TD error to priority
    beta = 0.4  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.  # clipped abs error

    def __init__(self, capacity):
        self.sumtree = SumTree(capacity)# 生成一个SumTree

    def store(self, transition):
        max_p = np.max(self.sumtree.tree[-self.sumtree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.sumtree.add(max_p, transition)   # set the max p for new p

    def sample(self, n):
        b_idx, b_memory, ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, self.sumtree.data[0].size)), np.empty((n, 1))
        pri_seg = self.sumtree.total_p / n       # priority segment
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1

        min_prob = np.min(self.sumtree.tree[-self.sumtree.capacity:]) / self.sumtree.total_p     # for later calculate ISweight
        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self.sumtree.get_leaf(v)
            prob = p / self.sumtree.total_p
            ISWeights[i, 0] = np.power(prob/min_prob, -self.beta)
            b_idx[i], b_memory[i, :] = idx, data
        return b_idx, b_memory, ISWeights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.sumtree.update(ti, p)

class Agent_Prioritized_Replay_DQN:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.005,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=500,
            memory_size=10000,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False,
            prioritized=True,
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

        self.prioritized = prioritized    # decide to use double q or not
        # 初始化记忆池
        if self.prioritized:
            self.memory = Memory(capacity=memory_size)
        else:
            self.memory = np.zeros((self.memory_size, n_features * 2 + 2))
            # self.memory = self.init_memory()
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
        # 存储历史检测数据
        self.cost_his = []  # 存储历史损失
        self.q_his = []  # 记录agent的动作价值
        self.running_q = 0



    def store_in_memory(self, s, a, r, s_):
        if self.prioritized:    # prioritized replay
            transition = np.hstack((s, [a, r], s_))
            self.memory.store(transition)    # have high priority for newly arrived transition
        else:       # random replay
            if not hasattr(self, 'memory_counter'):
                self.memory_counter = 0
            transition = np.hstack((s, [a, r], s_))
            index = self.memory_counter % self.memory_size
            self.memory[index, :] = transition
            self.memory_counter += 1

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

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('\ntarget_net的参数被更新\n')
        # 获取采样数据
        if self.prioritized:
            tree_idx, batch_memory, ISWeights = self.memory.sample(self.batch_size)
        else:
            batch_memory = self.pick_from_memory(batch_size=self.batch_size)

        q_next, q_eval = self.sess.run(
                [self.q_next, self.q_eval],
                feed_dict={self.s_: batch_memory[:, -self.n_features:],
                           self.s: batch_memory[:, :self.n_features]})

        q_target = q_eval.copy()
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        if self.prioritized:
            _, abs_errors, self.cost = self.sess.run([self._train_op, self.abs_errors, self.loss],
                                         feed_dict={self.s: batch_memory[:, :self.n_features],
                                                    self.q_target: q_target,
                                                    self.ISWeights: ISWeights})
            self.memory.batch_update(tree_idx, abs_errors)     # update priority
        else:
            _, self.cost = self.sess.run([self._train_op, self.loss],
                                         feed_dict={self.s: batch_memory[:, :self.n_features],
                                                    self.q_target: q_target})

        self.cost_his.append(self.cost)

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    # 初始化记忆池
    def init_memory(self):
        self.memory = np.zeros((self.memory_size, self.n_features * 2 + 2))


    # 从记忆池中取出一定数量的数据
    def pick_from_memory(self, batch_size):
        if self.memory_size < self.memory_counter:
            batch_indexs = np.random.choice(self.memory_size, size=batch_size)
        else:
            batch_indexs = np.random.choice(self.memory_counter, size=batch_size)
        batch_data = self.memory[batch_indexs, :]
        return batch_data

    # 初始化网络
    def init_eval_net(self):
        # 定义网络的输入输出
        self.s = tf.placeholder(dtype=tf.float32, shape=[None, self.n_features], name='s')
        self.q_target = tf.placeholder(dtype=tf.float32, shape=[None, self.n_actions], name='Q_target')
        if self.prioritized:
            self.ISWeights = tf.placeholder(tf.float32, [None, 1], name='IS_weights')
        with tf.variable_scope('eval_net'):
            # 定义神经层的配置
            n_Layer1 = 20
            c_names = ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            w_initializer = tf.random_normal_initializer(0., 0.3)
            b_initializer = tf.constant_initializer(0.1)
            self.q_eval = self.__create_fc_layer(self.s, n_Layer1, w_initializer, b_initializer, c_names,True)
            # 搭建损失函数和优化器
            with tf.variable_scope('loss'):
                if self.prioritized:
                    self.abs_errors = tf.reduce_sum(tf.abs(self.q_target - self.q_eval), axis=1)  # for updating Sumtree
                    self.loss = tf.reduce_mean(self.ISWeights * tf.squared_difference(self.q_target, self.q_eval))
                else:
                    self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
            with tf.variable_scope('train'):
                self._train_op = tf.train.RMSPropOptimizer(self.learn_rate).minimize(self.loss)
    def init_target_net(self):
        # 定义网络的输入输出
        self.s_ = tf.placeholder(dtype=tf.float32, shape=[None, self.n_features], name='s_')
        with tf.variable_scope('target_net'):
            # 定义神经层的配置
            n_Layer1 = 20
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            w_initializer = tf.random_normal_initializer(0., 0.3)
            b_initializer = tf.constant_initializer(0.1)

            self.q_next = self.__create_fc_layer(self.s_, n_Layer1, w_initializer, b_initializer, c_names,False)
    def __create_fc_layer(self, input, n_Layer, w_initializer, b_initializer, c_names,trainable):
        # 定义L1
        with tf.variable_scope('layer_1'):
            w1 = tf.get_variable(name='w1',
                                 shape=[self.n_features, n_Layer],
                                 initializer=w_initializer,
                                 collections=c_names,
                                 trainable=trainable)
            b1 = tf.get_variable(name='b1',
                                 shape=[1, n_Layer],
                                 initializer=b_initializer,
                                 collections=c_names,
                                 trainable=trainable)
            layer_1 = tf.nn.relu(tf.matmul(input, w1) + b1)
        # 定义L2
        with tf.variable_scope('layer_2'):
            w2 = tf.get_variable(name='w2',
                                 shape=[n_Layer, self.n_actions],
                                 initializer=w_initializer,
                                 collections=c_names,
                                 trainable=trainable)
            b2 = tf.get_variable(name='b2',
                                 shape=[1, self.n_actions],
                                 initializer=b_initializer,
                                 collections=c_names,
                                 trainable=trainable)
            out = tf.matmul(layer_1, w2) + b2
        return out
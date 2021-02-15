import  numpy as np
import tensorflow as tf

np.random.seed(1)
tf.set_random_seed(1)

# 使用DQN算法的Agent
class Agent_DQN:
    # 初始化参数
    def __init__(
            self,
            n_features,# 观测值个数
            n_actions, #动作个数
            learning_rate=0.01,# 学习率
            e_greedy=0.9,# e-greedy
            e_greedy_increment=None, #是否让greedy变化
            batch_size=32,# 每次采样数据的大小
            memory_size = 500, # 记忆池的行数据大小
            replace_target_iter=300,
            gamma=0.9, # 回报折扣因子
            output_graph = False, # 是否输出TensorBoard
    ):
        self.n_features = n_features
        self.n_actions = n_actions
        self.learn_rate =learning_rate
        self.epsilon_increment = None
        self.epsilon_increment = e_greedy_increment
        self.epsilon_max = e_greedy
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.batch_size = batch_size
        self.gamma = gamma
        self.memory_size = memory_size
        self.replace_target_iter=replace_target_iter
        # 初始化记忆池
        self.init_memory()
        # 初始化网络
        self.init_eval_net()
        self.init_target_net()
        # 定时复制参数给target_net
        self.learn_step_counter=0
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]
        # 初始化会话
        self.Session = tf.Session()
        # 是否输出图
        if output_graph:
            # $ tensorboard --logdir=logs
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.Session.graph)
        self.Session.run(tf.global_variables_initializer())
        # 存储损失
        self.cost_his = []
    # 选择动作(epsilon greedy)
    def choose_Action(self,s):
        # 将一维数组转换为二维数组，虽然只有一行
        s = s[np.newaxis,:]

        if np.random.uniform()<self.epsilon:
            action_values = self.Session.run( fetches=self.q_eval,feed_dict={self.s:s})
            action = np.argmax(action_values)
        else:
            action = np.random.randint(0,self.n_actions)
        return action
    # 学习策略
    def learn_from_step(self):
        # 检查是否复制参数给target_net
        self.learn_step_counter += 1
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.Session.run(self.replace_target_op)
            print('\ntarget_net的参数被更新\n')
        # 获取采样数据
        batch_data = self.pick_from_memory(batch_size=self.batch_size)
        # 像两个神经网络中输入观测值获取对应的动作价值，输出为行数为采样个数，列数为动作数的矩阵
        q_eval,q_next = self.Session.run(
            fetches=[self.q_eval,self.q_next],
            feed_dict = {
                self.s:batch_data[:,:self.n_features],
                self.s_:batch_data[:,-self.n_features:]
            })
        # 获取立即回报
        q_target = q_eval.copy()
        # 获取采样数据的索引，要修改的矩阵的行
        batch_index = np.arange(self.batch_size,dtype=np.int32)
        # 获取评估的动作的索引，要修改的矩阵的列
        eval_act_index = batch_data[:,self.n_features].astype(int)
        # 获取要修改Q值的立即回报
        reward = batch_data[:, self.n_features + 1]
        # 计算Q现实值，只修改矩阵中对应状态动作的Q值
        q_target[batch_index,eval_act_index] = reward+self.gamma*np.max(q_next,axis=1)
        # 进行优化
        _,self.cost = self.Session.run(fetches=[self.train_step,self.loss],
                                feed_dict={
                                    self.s: batch_data[:,:self.n_features],
                                self.q_target:q_target})
        # 存储损失值
        self.cost_his.append(self.cost)
        # 逐步提高的利用概率
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        return
    # 初始化记忆池
    def init_memory(self):
        self.memory_counter = 0
        self.memory = np.zeros((self.memory_size, self.n_features*2+2))

    # 向记忆池存入数据
    def store_in_memory(self,s,a,r,s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        # 对数据进行预处理
        transitions = np.hstack((s,[a,r],s_))

        insert_index = self.memory_counter % self.memory_size
        self.memory[insert_index,:]=transitions

        self.memory_counter+=1

    # 从记忆池中取出一定数量的数据
    def pick_from_memory(self,batch_size):
        if self.memory_size<self.memory_counter:
            batch_indexs = np.random.choice(self.memory_size,size=batch_size)
        else:
            batch_indexs = np.random.choice(self.memory_counter, size=batch_size)
        batch_data = self.memory[batch_indexs,:]
        return batch_data

    # 初始化网络
    def init_eval_net(self):
        # 定义网络的输入输出
        self.s = tf.placeholder(dtype=tf.float32,shape= [None,self.n_features],name='s')
        self.q_target = tf.placeholder(dtype=tf.float32,shape=[None,self.n_actions],name='Q_target')
        with tf.variable_scope('eval_net'):
            # 定义神经层的配置
            n_Layer1=10
            c_names=['eval_net_params',tf.GraphKeys.GLOBAL_VARIABLES]
            w_initializer = tf.random_normal_initializer(0.,0.3)
            b_initializer = tf.constant_initializer(0.1)
            # 定义L1
            with tf.variable_scope('layer_1'):
                w1 = tf.get_variable(name='w1',
                                     shape=[self.n_features,n_Layer1],
                                     initializer=w_initializer,
                                     collections=c_names)
                b1 = tf.get_variable(name='b1',
                                     shape=[1,n_Layer1],
                                     initializer=b_initializer,
                                     collections=c_names)
                layer_1 = tf.nn.relu(tf.matmul(self.s,w1)+b1)
            # 定义L2
            with tf.variable_scope('layer_2'):
                w2 = tf.get_variable(name='w2',
                                     shape=[n_Layer1, self.n_actions],
                                     initializer=w_initializer,
                                     collections=c_names)
                b2 = tf.get_variable(name='b2',
                                     shape=[1, self.n_actions],
                                     initializer=b_initializer,
                                     collections=c_names)
                self.q_eval = tf.matmul(layer_1, w2) + b2
            #搭建损失函数和优化器
            with tf.variable_scope('loss'):
                self.loss = tf.reduce_mean(tf.squared_difference(self.q_target,self.q_eval))
            with tf.variable_scope('train'):
                self.train_step = tf.train.RMSPropOptimizer(self.learn_rate).minimize(self.loss)
    def init_target_net(self):
        # 定义网络的输入输出
        self.s_ = tf.placeholder(dtype=tf.float32,shape= [None,self.n_features],name='s_')
        with tf.variable_scope('target_net'):
            # 定义神经层的配置
            n_Layer1=10
            c_names=['target_net_params',tf.GraphKeys.GLOBAL_VARIABLES]
            w_initializer = tf.random_normal_initializer(0.,0.3)
            b_initializer = tf.constant_initializer(0.1)
            # 定义L1
            with tf.variable_scope('layer_1'):
                w1 = tf.get_variable(name='w1',
                                     shape=[self.n_features,n_Layer1],
                                     initializer=w_initializer,
                                     collections=c_names)
                b1 = tf.get_variable(name='b1',
                                     shape=[1,n_Layer1],
                                     initializer=b_initializer,
                                     collections=c_names)
                layer_1 = tf.nn.relu(tf.matmul(self.s_,w1)+b1)
            # 定义L2
            with tf.variable_scope('layer_2'):
                w2 = tf.get_variable(name='w2',
                                     shape=[n_Layer1, self.n_actions],
                                     initializer=w_initializer,
                                     collections=c_names)
                b2 = tf.get_variable(name='b2',
                                     shape=[1, self.n_actions],
                                     initializer=b_initializer,
                                     collections=c_names)
                self.q_next = tf.matmul(layer_1, w2) + b2
    # 显示图
    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()



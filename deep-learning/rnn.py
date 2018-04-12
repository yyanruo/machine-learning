import numpy as np
from cnn import Reluactivator,IdentityActivator,element_wise_op
from functools import reduce
class Recurrentlayer(object):
    def __init__(self, input_w, state_w, activator, learning_rate):
        self.input_w = input_w
        self.state_w = state_w
        self.activator = activator
        self.learning_rate = learning_rate
        self.times = 0  # 当前时刻初始化为t0
        self.state_list = []  # 保存各个时刻的state
        self.state_list.append(np.zeros((state_w, 1)))
        self.U = np.random.uniform(-1e-4, 1e-4, (state_w, input_w))
        self.W = np.random.uniform(-1e-4, 1e-4, (state_w, state_w))

    def forward(self, input_array):
        """
        向前传播 : St = f(U * Xt + W * St-1)
        """
        self.times += 1
        state = (np.dot(self.U, input_array) + np.dot(self.W, self.state_list[-1]))
        element_wise_op(state, self.activator.forward)
        self.state_list.append(state)

    def backward(self, sensitivity_array, activator):
        """
        反向传播 
        """
        self.calc_delta(sensitivity_array, activator)
        self.calc_gradient()

    def calc_delta(self, sensitivity_array, activator):
        """
        误差传播 : 计算时间t范围内所有的误差
        """
        self.delta_list = []
        for i in range(self.times):
            self.delta_list.append(np.zeros((self.state_w,1)))
        self.delta_list.append(sensitivity_array)
        # 从后往前算 dt
        for k in range(self.times-1, 0, -1):
            self.calc_delta_k(k, activator)

    def calc_delta_k(self,k, activator):
        """
        误差传播 : dt-1.T = dt.T * W * diag(f'(St))
        """
        state = self.state_list[k+1].copy()
        element_wise_op(self.state_list[k + 1], activator.backward)  #f'(St)
        self.delta_list[k] = np.dot(np.dot(self.delta_list[k+1].T, self.W), np.diag(state[:,0])).T

    def calc_gradient(self):
        """
        梯度计算 : 
        """
        self.gradient_list = []
        for t in range(self.times + 1):
            self.gradient_list.append(np.zeros((self.state_w, self.state_w)))
        for t in range(self.times, 0, -1):
            self.calc_gradient_t(t)
        # 各个时间梯度之和
        self.gradient = reduce(lambda a,b: a + b, self.gradient_list, self.gradient_list[0])

    def calc_gradient_t(self, t):
        gradient = np.dot(self.delta_list[t], self.state_list[t-1].T)
        self.gradient_list[t] = gradient

    def update(self):
        """
        权值更新
        """
        self.W -= self.learning_rate * self.gradient

    def reset_state(self):
        #重置循环层
        self.times = 0
        self. state_list = []
        self.state_list.append(np.zeros((self.state_w, 1)))

def gradient_check():
    """
    梯度检测
    """
    # 设计一个误差函数，取所有节点输出项之和
    error_function = lambda o: o.sum()
    rl = Recurrentlayer(3, 2,IdentityActivator(), 1e-3)

    x = [np.array([[1], [2], [3]]), np.array([[2], [3], [4]])]
    d = np.array([[1], [2]])
    rl.forward(x[0])
    rl.forward(x[1])
    sensitivity_array = np.ones(rl.state_list[-1].shape, dtype=np.float64)
    # 计算梯度
    rl.backward(sensitivity_array, IdentityActivator())

    epslion = 10e-4
    for i in range(rl.W.shape[0]):
        for j in range(rl.W.shape[1]):
            rl.W[i,j] += epslion
            rl.reset_state()
            rl.forward(x[0])
            rl.forward(x[1])
            error1 = error_function(rl.state_list[-1])
            rl.W[i,j] -= 2 * epslion
            rl.reset_state()
            rl.forward(x[0])
            rl.forward(x[1])
            error2 = error_function(rl.state_list[-1])
            expect_grad = (error1 - error2) / (2 * epslion)
            rl.W[i,j] += epslion

            print('weights(%d,%d): expected - actural %f - %f' %
                  (i, j, expect_grad, rl.gradient[i, j]))


def test():
    l = Recurrentlayer(3, 2, Reluactivator(), 1e-3)
    x = [np.array([[1], [2], [3]]), np.array([[2], [3], [4]])]
    d = np.array([[1], [2]])
    l.forward(x[0])
    l.forward(x[1])
    l.backward(d, Reluactivator())
    return l
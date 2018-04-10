import numpy as np

class Reluactivator(object):
    """
    RELU
    """
    def forward(self, weighted_input):
        return max(0, weighted_input)
    def backward(self, output):
        return 1 if output > 0 else 0

def conv(input_array, kernel_array, output_array, stride, bias):
    """
    卷积操作
    Args:
        input_array : [输入二维数组]
        kernel_array : [卷积核]
        output_array : [输出二维数组，需要self操作]
        stride : [步长]
        bias : [偏差]
    """
    output_w = output_array.shape[1]
    output_h = output_array.shape[0]
    kernel_w = kernel_array.shape[-1]
    kernel_h = kernel_array.shape[-2]
    for i in range(output_w):
        for j in range(output_h):
            # A = W * X + b
            output_array[i][j] = (get_patch(input_array, i, j, kernel_w, kernel_h, stride)
                      * kernel_array).sum() + bias

def get_patch(input_array, i, j, filter_w, filter_h, stride):
    """
    获取卷积区域
    Args:
        input_array : [输入二维数组]
        i : [输出二维数组的中的W]
        j : [输出二维数组的中的H]
        filter_w : [滤波器的W]
        filter_h : [滤波器的H]
        stride : [步长]
    Returns:
        [卷积小矩阵]: [输出矩阵中i,j位置对应在输入矩阵中的卷积区域矩阵]
    """
    start_i = i * stride
    start_j = j * stride
    if input_array.ndim == 2:
        return input_array[start_i:start_i + filter_h, start_j:
                           start_j + filter_w]
    elif input_array.ndim == 3:
        return input_array[:, start_i:start_i + filter_h, start_j:
                           start_j + filter_w]

def get_max_index(array):
    """
    获取一个2D区域的最大值所在的索引
    """
    max_i = 0
    max_j = 0
    max_value = array[0, 0]
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if array[i, j] > max_value:
                max_value = array[i, j]
                max_i, max_j = i, j
    return max_i, max_j


def padding(input_array, zp):
    """
    对输入矩阵进行zp填充
    """
    if zp == 0:
        return input_array
    else:
        if input_array.ndim == 3:
            input_w = input_array.shape[2]
            input_h = input_array.shape[1]
            input_d = input_array.shape[0]
            padded_array = np.zeros((input_d, input_h + 2 * zp,
                                     input_w + 2 * zp))
            padded_array[:, zp:zp + input_h, zp:zp + input_w] = input_array
            return padded_array
        elif input_array.ndim == 2:
            input_w = input_array.shape[1]
            input_h = input_array.shape[0]
            padded_array = np.zeros((input_h + 2 * zp, input_w + 2 * zp))
            padded_array[zp:zp + input_h, zp:zp + input_w] = input_array
            return padded_array


def element_wise_op(array, op):
    """
    对矩阵每个元素进行op操作
    """
    for i in np.nditer(array, op_flags=['readwrite']):
        i[...] = op(i)


class Convlayer(object):
    def __init__(self, input_w, input_h, input_d, filter_w, filter_h,
                 filter_num, zero_padding, stride, activator, learning_rate):
        self.input_w = input_w
        self.input_h = input_h
        self.input_d = input_d
        self.filter_w = filter_w
        self.filter_h = filter_h
        self.filter_num = filter_num
        self.zero_padding = zero_padding
        self.stride = stride
        self.output_w = Convlayer.calculate_output_size(self.input_w, filter_w, zero_padding, stride)
        self.output_h = Convlayer.calculate_output_size(self.input_h, filter_h, zero_padding, stride)
        self.output_array = np.zeros((self.filter_num, self.output_h,self.output_w))
        self.filters = []
        # 添加Filter
        for i in range(filter_num):
            self.filters.append(Filter(filter_w, filter_h, self.filter_num))
        self.activator = activator
        self.learning_rate = learning_rate

    def forward(self, input_array):
        """
        向前传播
        """
        self.input_array = input_array
        self.padded_input_array = padding(input_array, self.zero_padding)
        for f in range(self.filter_num):
            filter = self.filters[f]
            conv(self.padded_input_array, filter.get_weights(),
                 self.output_array[f], self.stride, filter.get_bias())
        element_wise_op(self.output_array, self.activator.forward)

    def backward(self, input_array, sensitivity_array, activator):
        """
        反向传播
        """
        self.forward(input_array)
        self.bp_sensitivity_map(sensitivity_array, activator)
        self.bp_gradient(sensitivity_array)

    def bp_sensitivity_map(self, sensitivity_array, activator):
        """
        误差反传
        """
        #处理卷积步长，还原为S=1的情况
        expanded_array = self.expend_sensitivity_map(sensitivity_array)
        #zero-padding
        expanded_w = expanded_array.shape[2]
        zp = int((self.input_w + self.filter_w - 1 - expanded_w) / 2)
        padded_array = padding(expanded_array, zp)
        #delta_array用于保存反传到上一层的激活map
        self.delta_array = np.zeros((self.input_d, self.input_h, self.input_w))
        for f in range(self.filter_num):
            filter = self.filters[f]
            #旋转180度卷积核权值矩阵
            flipped_weights = np.array(
                list(map(lambda i: np.rot90(i,2), filter.get_weights())))
            delta_array = np.zeros((self.input_d, self.input_h, self.input_w))
            for d in range(delta_array.shape[0]):
                conv(padded_array[f], flipped_weights[d], delta_array[d], 1, 0)
            self.delta_array += delta_array
        #激活函数的偏导数
        derivative_array = np.array(self.input_array)
        element_wise_op(derivative_array, activator.backward)
        self.delta_array += derivative_array

    def expend_sensitivity_map(self, sensitivity_array):
        """
        处理卷积步长，还原为S=1的情况
        """
        depth = sensitivity_array.shape[0]
        expanded_w = (self.input_w - self.filter_w + 2 * self.zero_padding + 1)
        expanded_h = (self.input_h - self.filter_h + 2 * self.zero_padding + 1)
        expand_array = np.zeros((depth, expanded_h, expanded_w))
        for i in range(self.output_h):
            for j in range(self.output_w):
                i_pos = i * self.stride
                j_pos = j * self.stride
                expand_array[:, i_pos, j_pos] = sensitivity_array[:, i, j]
        return expand_array

    def bp_gradient(self, sensitivity_array):
        """
        计算权值和bias的梯度
        """
        expanded_array = self.expend_sensitivity_map(sensitivity_array)
        for f in range(self.filter_num):
            filter = self.filters[f]
            for d in range(filter.weights.shape[0]-1):
                conv(self.padded_input_array[d], expanded_array[f],
                     filter.w_grad[d], 1, 0)
            filter.b_grad = expanded_array[f].sum()

    def update(self):
        for filter in self.filters:
            filter.update(self.learning_rate)

    @staticmethod
    def calculate_output_size(input_size, filter_size, zero_padding, stride):
        return int((input_size - filter_size + 2 * zero_padding) / stride + 1)


class Filter(object):
    def __init__(self, w, h, d):
        self.weights = np.random.uniform(-1e-4, 1e-4, (d, h, w))
        self.b = 0
        self.w_grad = np.zeros(self.weights.shape)
        self.b_grad = 0

    def __repr__(self):
        return 'filter weights:\n%s\nbias:\n%s' % (repr(self.weights),
                                                   repr(self.b))

    def get_weights(self):
        return self.weights

    def get_bias(self):
        return self.b

    def update(self, learning_rate):
        self.weights -= learning_rate * self.w_grad
        self.b -= learning_rate * self.b


class MaxPoolinglayer(object):
    def __init__(self, input_w, input_h, input_d, filter_w, filter_h, stride):
        self.input_w = input_w
        self.input_h = input_h
        self.input_d = input_d
        self.filter_w = filter_w
        self.filter_h = filter_h
        self.stride = stride
        self.output_w = (input_w - filter_w) / self.stride + 1
        self.output_h = (input_h - filter_h) / self.stride + 1
        self.output_array = np.zeros((self.input_d, self.output_h,
                                      self.output_w))

    def forward(self, input_array):
        for d in range(self.input_d):
            for i in range(self.input_h):
                for j in range(self.input_w):
                    self.output_array[d, i, j] = (get_patch(
                        input_array[d], i, j, self.filter_w, self.filter_h,
                        self.stride).max())

    def backward(self, input_array, sensitivity_array):
        self.delta_array = np.zeros(input_array.shape)
        for d in range(self.input_d):
            for i in range(self.output_h):
                for j in range(self.output_w):
                    patch_array = get_patch(input_array[d], i, j,
                                            self.filter_w, self.filter_h,
                                            self.stride)
                    k, l = get_max_index(patch_array)
                    self.delta_array[d, i * self.stride + k, j * self.stride +
                                     l] = sensitivity_array[d, i, j]


class IdentityActivator(object):
    def forward(self, weighted_input):
        return weighted_input

    def backward(self, output):
        return 1


def init_test():
    a = np.array(
        [[[0, 1, 1, 0, 2], [2, 2, 2, 2, 1], [1, 0, 0, 2, 0], [0, 1, 1, 0, 0],
          [1, 2, 0, 0, 2]], [[1, 0, 2, 2, 0], [0, 0, 0, 2, 0], [1, 2, 1, 2, 1],
                             [1, 0, 0, 0, 0], [1, 2, 1, 1, 1]],
         [[2, 1, 2, 0, 0], [1, 0, 0, 1, 0], [0, 2, 1, 0, 1], [0, 1, 2, 2, 2],
          [2, 1, 0, 0, 1]]])
    b = np.array([[[0, 1, 1], [2, 2, 2], [1, 0, 0]], [[1, 0, 2], [0, 0, 0],
                                                      [1, 2, 1]]])
    cl = Convlayer(5, 5, 3, 3, 3, 2, 1, 2, IdentityActivator(), 0.001)
    cl.filters[0].weights = np.array(
        [[[-1, 1, 0], [0, 1, 0], [0, 1, 1]], [[-1, -1, 0], [0, 0, 0], [
            0, -1, 0
        ]], [[0, 0, -1], [0, 1, 0], [1, -1, -1]]],
        dtype= np.float32)
    cl.filters[0].bias = 1
    cl.filters[1].weights = np.array(
        [[[1, 1, -1], [-1, -1, 1], [0, -1, 1]], [[0, 1, 0], [-1, 0, -1], [
            -1, 1, 0
        ]], [[-1, 0, 0], [-1, 0, 1], [-1, 0, 0]]],
        dtype= np.float32)
    return a, b, cl


def gradient_check():
    '''
    梯度检查
    '''
    # 设计一个误差函数，取所有节点输出项之和
    error_function = lambda o: o.sum()
    # 计算forward值
    a, b, cl = init_test()
    cl.forward(a)
    # 求取sensitivity map，是一个全1数组
    sensitivity_array = np.ones(cl.output_array.shape, dtype = np.float32)
    # 计算梯度
    cl.backward(a, sensitivity_array, IdentityActivator())
    # 检查梯度
    epsilon = 10e-4
    for d in range(cl.filters[0].w_grad.shape[0]):
        for i in range(cl.filters[0].w_grad.shape[1]):
            for j in range(cl.filters[0].w_grad.shape[2]):
                cl.filters[0].weights[d, i, j] += epsilon
                cl.forward(a)
                err1 = error_function(cl.output_array)
                cl.filters[0].weights[d, i, j] -= 2 * epsilon
                cl.forward(a)
                err2 = error_function(cl.output_array)
                expect_grad = (err1 - err2) / (2 * epsilon)
                cl.filters[0].weights[d, i, j] += epsilon
                print('weights(%d,%d,%d): expected - actural %f - %f' % (
                    d, i, j, expect_grad, cl.filters[0].w_grad[d, i, j]))

if __name__ == '__main__':
    gradient_check()
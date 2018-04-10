import random
import numpy as np


class IdentityActivator(object):
    def forward(self, weighted_input):
        return weighted_input

    def backward(self, output):
        return 1


class SigmoidActivator(object):
    def forward(self, weighted_input):
        return 1.0 / (1.0 + np.exp(-weighted_input))

    def backward(self, output):
        return output * (1 - output)


class FClayer(object):
    def __init__(self, input_size, output_size, activator):
        """
        构造函数
        input_size: [本层输入向量的维度]
        output_size: [本层输出向量的维度]
        activator: [激活函数]
        """
        self.input_size = input_size
        self.output_size = output_size
        self.activator = activator

        self.W = np.random.uniform(-0.1, 0.1, (output_size, input_size))
        self.b = np.zeros((output_size, 1))
        self.output = np.zeros((output_size, 1))

    def forward(self, input_array):
        """
        向前传播
        input_array: [输入向量，维度必须等于input_size]
        """
        self.input = input_array
        self.output = self.activator.forward(
            np.dot(self.W, input_array) + self.b)

    def backword(self, delta_array):
        """
        反向传播
        delta_array: [上层的误差项]
        """
        self.delta = self.activator.backword(self.output) * np.dot(
            self.W.T, delta_array)
        self.W_grad = np.dot(delta_array, self.input.T)
        self.b_grad = delta_array

    def update(self, learning_rate):
        """
        更新函数
        learning_rate: [学习率]
        """
        self.W += learning_rate * self.W_grad
        self.b += learning_rate * self.b_grad

    def dump(self):
        print('W:%s\nb:%s' % (self.W, self.b))


class Network(object):
    def __init__(self, layers):
        self.layers = []
        for i in range(len(layers) - 1):
            self.layers.append(
                FClayer(layers[i], layers[i + 1], SigmoidActivator()))

    def predict(self, sample):
        """
        预函数测
        sample:[输入样本]
        """
        output = sample
        for layer in self.layers:
            layer.forward(output)
            output = layer.output
        return output

    def train(self, labels, data_set, rate, epoch):
        """
        训练函数
        labels:[样本标签]
        data_set:[输入样本集]
        rata:[学习率]
        epoch:[训练次数]
        """
        for i in range(epoch):
            for d in range(len(data_set)):
                self.train_one_sample(labels[d], data_set[d], rate)

    def train_one_sample(self, label, sample, rate):
        self.predict(sample)
        self.calc_gradient(label)
        self.update_weight(rate)

    def calc_gradient(self, label):
        delta = self.layers[-1].activator.backward(self.layers[-1].output) * (
            label - self.layers[-1].output)
        for layer in self.layers[::-1]:
            layer.backward(delta)
            delta = layer.delta
        return delta

    def update_weight(self, rate):
        for layer in self.layers:
            layer.update(rate)

    def dump(self):
        for layer in self.layers:
            layer.dump()

    def loss(self, output, label):
        return 0.5 * ((label - output) * (label - output)).sum()

    def gradient_check(self, sampel_feature, sample_label):
        """
        梯度检查
        sampel_feature: [输入样本]
        sample_label: [输入标签]
        """

        self.predict(sampel_feature)
        self.calc_gradient(sample_label)

        epsilon = 10e-4
        for fc in self.layers:
            for i in range(fc.W.shape[0]):
                for j in range(fc.W.shape[1]):
                    fc.W[i,j] += epsilon
                    output = self.predict(sampel_feature)
                    error1 = self.loss(sample_label, output)
                    fc.W[i,j] += epsilon
                    output = self.predict(sampel_feature)
                    error2 = self.loss(sampel_feature, output)
                    expect_grad = (error1- error2) / (2 * epsilon)
                    fc.W[i,j] += epsilon
                    print('W(%d,%d):expecded - actural %.4e - %.4e' % (i, j, expect_grad, fc.W_grad[i,j]))

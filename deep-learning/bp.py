from functools import reduce
import random

# 节点类，记录与维护自身已经节点相关的上下层连接，实现输出值与误差计算
class Node(object):
    #构造节点对象：节点所属层、节点编号、下层连接、上层连接、输出、误差
    def __init__(self, layer_index, node_index):
        self.layer_index = layer_index
        self.node_index = node_index
        self.nextlayer = []
        self.prelayer = []
        self.output = 0
        self.delta = 0

    def set_output(self, output):
        self.output = output

    #添加一个到下层节点的连接
    def append_nextlayer_connection(self, conn):
        self.nextlayer.append(conn)

    #添加一个到上层节点的连接
    def append_prelayer_connection(self, conn):
        self.prelayer.append(conn)

    #计算节点的输出
    def calc_output(self):
        output = reduce(
            lambda ret, conn: ret + conn.prelayer_node.output * conn.weight,
            self.prelayer, 0)
        self.output = sigmoid(output)

    #如果节点属于隐藏层，计算delta
    def calc_hidden_layer_delta(self):
        nextlayer_delta = reduce(
            lambda ret, conn: ret + conn.nextlayer_node.delta * conn.weight,
            self.nextlayer, 0.0)
        self.delta = self.output * (1 - self.output) * nextlayer_delta

    #如果节点属于输出层，计算delta
    def calc_output_layer_delta(self, label):
        self.delta = self.output * (1 - self.output) * (label - self.output)

    #打印节点信息
    def __str__(self):
        node_str = '%u-%u:output:%f delta:%f' % (self.layer_index,
                                                 self.node_index, self.output,
                                                 self.delta)
        nextlayer_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn),
                               self.nextlayer, '')
        prelayer_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn),
                              self.prelayer, '')
        return node_str + '\n\tnextlayer:' + nextlayer_str + '\n\tprelayer:' + prelayer_str


## 构造一个ConstNode对象，为了实现一个输出恒为1的节点
class ConstNode(object):
    def __init__(self, layer_index, node_index):
        self.layer_index = layer_index
        self.node_index = node_index
        self.nextlayer = []
        self.output = 1

    #添加一个到下层节点的连接
    def append_nextlayer_connection(self, conn):
        self.nextlayer.append(conn)

    #如果节点属于隐藏层，计算delta
    def calc_hidden_layer_delta(self):
        nextlayer_delta = reduce(
            lambda ret, conn: ret + conn.nextlayer_node.delta * conn.weight,
            self.nextlayer, 0.0)
        self.delta = self.output * (1 - self.output) * nextlayer_delta

    #打印节点信息
    def __str__(self):
        node_str = '%u-%u:output:1' % (self.layer_index, self.node_index)
        nextlayer_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn),
                               self.nextlayer, '')
        return node_str + '\n\tnextlayer:' + nextlayer_str


## Layer对象，初始化一层，每层最后一个节点为ConstNode
class Layer(object):
    #初始化：层编号、层包含节点个数
    def __init__(self, layer_index, node_count):
        self.layer_index = layer_index
        self.nodes = []
        for i in range(node_count):
            self.nodes.append(Node(layer_index, i))
        self.nodes.append(ConstNode(layer_index, node_count))  #最后一个为ConstNode

    #输入层设置节点输出
    def set_output(self, data):
        for i in range(len(data)):
            self.nodes[i].set_output(data[i])

    #计算层的输出向量
    def calc_output(self):
        for node in self.nodes[:-1]:
            node.calc_output()

    #打印层信息
    def dump(self):
        for node in self.nodes:
            print(node)


## conneciton对象，记录连接的权重，以及其上下游节点
class connection(object):
    def __init__(self, prelayer_node, nextlayer_node):
        self.prelayer_node = prelayer_node
        self.nextlayer_node = nextlayer_node
        self.weight = random.uniform(-0.1, 0.1)
        self.gradient = 0.0

    #计算梯度
    def calc_gradient(self):
        self.gradient = self.nextlayer_node.delta * self.prelayer_node.output

    #获取梯度
    def get_gradient(self):
        return self.gradient

    #梯度下降更新权重
    def update_weight(self, rate):
        self.calc_gradient()
        self.weight += rate * self.gradient

    #打印连接信息
    def __str__(self):
        return '(%u-%u)->(%u-%u) = %f' % (self.prelayer_node.layer_index,
                                          self.prelayer_node.node_index,
                                          self.nextlayer_node.layer_index,
                                          self.nextlayer_node.node_index,
                                          self.weight)


## Connections对象，提供Connection集合
class Connections(object):
    def __init__(self):
        self.connections = []

    def add_connection(self, connection):
        self.connections.append(connection)
    def dump(self):
        for conn in self.connections:
            print(conn)


## Network对象，提供API
class Network(object):
    #初始化一个全连接神经网络，输入layers为一个二维数组
    def __init__(self, layers):
        self.connections = Connections()
        self.layers = []
        layer_count = len(layers)
        node_count = 0
        #创建layers
        for i in range(layer_count):
            self.layers.append(Layer(i, layers[i]))
        #创建每一层的全连接conn
        for layer in range(layer_count - 1):
            connections = [
                connection(prelayer_node, nextlayer_node)
                for prelayer_node in self.layers[layer].nodes
                for nextlayer_node in self.layers[layer + 1].nodes[:-1]
            ]
            #添加每一层conn到self.connections中
            for conn in connections:
                self.connections.add_connection(conn)
                conn.nextlayer_node.append_prelayer_connection(conn)
                conn.prelayer_node.append_nextlayer_connection(conn)

    #训练：data_set（二维数组）
    def train(self, labels, data_set, rate, iteration):
        for i in range(iteration):
            for d in range(len(data_set)):
                self.train_one_sample(labels[d], data_set[d], rate)

    #训练一个样本
    def train_one_sample(self, label, sample, rate):
        self.predict(sample)
        self.calc_delta(label)
        self.update_weight(rate)

    #向前传播
    def predict(self, sample):
        self.layers[0].set_output(sample)
        for i in range(1, len(self.layers)):
            self.layers[i].calc_output()
        return map(lambda node: node.output, self.layers[-1].nodes[:-1])

    #反向传播
    def calc_delta(self, label):
        output_nodes = self.layers[-1].nodes
        #输出层的delta
        for i in range(len(label)):
            output_nodes[i].calc_output_layer_delta(label[i])
        #隐藏层的delta
        for layer in self.layers[-2::-1]:
            for node in layer.nodes:
                node.calc_hidden_layer_delta()

    #权值更新
    def update_weight(self, rate):
        for layer in self.layers[:-1]:
            for node in layer.nodes:
                for conn in node.nextlayer:
                    conn.update_weight(rate)

    #更新连接上的梯度
    def calc_gradient(self):
        for layer in self.layers[:-1]:
            for node in layer.nodes:
                for conn in node.nextlayer:
                    conn.calc_gradient()

    #获得网络在一个样本下，每个连接上的梯度
    def get_gradient(self, label, sample):
        self.predict(sample)
        self.calc_delta(label)
        self.calc_gradient()

    def dump(self):
        for layer in self.layers:
            layer.dump()


def gradient_check(network, sample_feature, sample_label):
    #计算网络误差
    network_error = lambda vec1, vec2: 0.5 * reduce(lambda a, b: a + b, map(lambda v: (v[0]-v[1])*(v[0]-v[1]), zip(vec1,vec2)))
    network.get_gradient(sample_feature, sample_label)

    for conn in network.connections.connections:
        actual_gradient = conn.get_gradient()

        epsilon = 0.0001
        conn.weight += epsilon
        error1 = network_error(network.predict(sample_feature),sample_label)

        conn.weight -= 2*epsilon
        error2 = network_error(network.predict(sample_feature),sample_label)
        
        expected_gradient = (error2 - error1) / (2 * epsilon)

        print('expected-g:\t%f\nacutral-g:\t%f'%(expected_gradient, actual_gradient))
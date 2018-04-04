class Perception(object):
    #初始化感知器（激活函数、权值、bais）
    def __init__(self, input_num, activator):
        self.activator = activator
        self.weights = [0.0 for _ in range(input_num)]
        self.bias = 0

    #print 权值与bias
    def __str__(self):
        return 'weight\t:%s\nbias\t:%s'%(self.weights,self.bias)

    #预测：输入向量输出计算结果
    def predict(self, input_vec):
        #[(x1,w1),...]->[x1*w1,...]->reduce求和
        return self.activator(reduce(lambda a,b: a + b,
                                    list(map(lambda x: x[0] * x[1], zip(input_vec, self.weights)))
                                    , 0.0) + self.bias)

    # 训练：输入向量
    def train(self, input_vecs, labels, iteration, lr):
        for i in range(iteration):
            self._one_iteration(input_vecs, labels, lr)

    #单次训练
    def _one_iteration(self, input_vecs, labels, lr):
        samples = zip(input_vecs, labels)
        for (input_vec, label) in samples:
            output = self.predict(input_vec)
            self._update_weights(input_vec, output, label, lr)
    #权值更新
    def _update_weights(self, input_vec, output, label, lr):
        delta = label - output
        
        self.weights = list(map(
            lambda x: x[1] + lr * delta * x[0],
            zip(input_vec, self.weights)))
        self.bias += lr * delta

#################################################################
from functools import reduce
#激活函数
def f(x):
    return 1 if x > 0 else 0

def get_training_dataset():
    #真值表
    input_vecs = [[1,1],[0,0],[1,0],[0,1]]
    labels = [1, 0, 0, 0]
    return input_vecs, labels

def train_and_perception():
    #2是因为and是二元函数
    p = Perception(2, f)
    input_vecs, labels = get_training_dataset()
    p.train(input_vecs, labels, 10, 0.1)

    return p

if __name__ == '__main__':
    and_perception = train_and_perception()
    print(and_perception)

    print('1 and 1 = %d' % and_perception.predict([1, 1]))
    print('0 and 0 = %d' % and_perception.predict([0, 0]))
    print('1 and 0 = %d' % and_perception.predict([1, 0]))
    print('0 and 1 = %d' % and_perception.predict([0, 1]))
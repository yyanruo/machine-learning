import glob
import os.path
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
import tensorflow.contrib.slim as slim

import tensorflow.contrib.slim.python.slim.nets.inception_v3 as inception_v3

INPUT_DATA = '/path/to/save_model'
TRAIN_FILE = 'train_dir/model'
CKPT_FILE = '/path/to/inception_v3'

# 定义训练中使用的参数。
LEARNING_RATE = 0.01
STEPS = 5000
BATCH = 128
N_CLASSES = 5

# 不需要从谷歌训练好的模型中加载的参数。这里就是最后的全联接层，因为在
# 新的问题中我们要重新训练这一层中的参数。这里给出的是参数的前缀。
CHECKPOINT_EXCLUDE_SCOPES = 'InceptionV3/Logits,InceptionV3/AuxLogits'
# 需要训练的网络层参数明层，在fine-tuning的过程中就是最后的全联接层。
# 这里给出的是参数的前缀。
TRAINABLE_SCOPES = 'InceptionV3/Logits'

# 获取所有需要从谷歌训练好的模型中加载的参数
def get_tuned_variables():
    exclusions = [scope.strip() for scope in CHECKPOINT_EXCLUDE_SCOPES.split(',')]

    variables_to_restore = []
    # 枚举inception-v3模型中所有的参数，然后判断是否需要从加载列表中移除
    for var in slim.get_model_variables():
        print(var.op.name)
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)
    return variables_to_restore


# 获取所有需要训练的变量列表。
def get_trainable_variable():
    scopes = [scope.strip() for scope in TRAINABLE_SCOPES.split(',')]
    variables_to_train = []
    # 枚举所有需要训练的参数前缀，并通过这些前缀找到所有的参数。
    for scope in scopes:
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        variables_to_train.extend(variables)
    return variables_to_train

def main():
    # 加载预处理好的数据。
    processed_data = np.load(INPUT_DATA)
    training_images = processed_data[0]
    n_training_example = len(training_images)
    training_labels = processed_data[1]
    validation_images = processed_data[2]
    validation_labels = processed_data[3]
    testing_images = processed_data[4]
    testing_labels = processed_data[5]

    # 定义inception-v3的输入，images为输入图片，labels为每一张图片对应的标签。
    images = tf.placeholder(
        tf.float32, [None, 299, 299, 3], name='input_images')
    labels = tf.placeholder(tf.int64, [None], name='labels')
    # 定义inception-v3模型。使用了dropout，需要定义训练时和测试时的模型。
    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        train_logits, _ = inception_v3.inception_v3(
            images, num_classes=N_CLASSES, is_training=True)
        test_logits, _ = inception_v3.inception_v3(
            images, num_classes=N_CLASSES, is_training=True,reuse=True)

    trainable_variabels = get_trainable_variable()
    print(trainable_variabels)
    # 计算正确率。
    cross_entrop = tf.nn.softmax_cross_entropy_with_logits(
        logits=train_logits,
        labels=tf.one_hot(labels, N_CLASSES))
    cross_entrop_mean = tf.reduce_mean(cross_entrop)
    train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(
        cross_entrop_mean, var_list=trainable_variabels)

    with tf.name_scope('evalution'):
        correct_prediction = tf.equal(tf.arg_max(test_logits,1), labels)
        evaulation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # 加载谷歌已经训练好的模型。
    loader = tf.train.Saver(get_tuned_variables())
    saver = tf.train.Saver()
    with tf.variable_scope("InceptionV3", reuse=True):
        check1 = tf.get_variable("Conv2d_1a_3x3/weights")
        check2 = tf.get_variable("Logits/Conv2d_1c_1x1/weights")

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        print(sess.run(check1))
        print(sess.run(check2))
        print('Loading tured model from %s' % CKPT_FILE)
        loader.restore(sess, CKPT_FILE)

        start = 0
        end =BATCH
        for i in range(STEPS):
            print(sess.run(check1))
            print(sess.run(check2))
            _, loss = sess.run([train_step, cross_entrop_mean], feed_dict={
                images: training_images[start:end],
                labels: training_labels[start:end]})
            if i % 100 == 0 or i + 1 == STEPS:
                saver.save(sess, TRAIN_FILE, global_step=i)
                validation_accuracy = sess.run(evaulation_step, feed_dict={
                    images: validation_images, labels:validation_labels})
                print('Step %d: Training loss is %.1f%% Validation accuracy = %.1f%%' % (
                    i, loss * 100.0, validation_accuracy * 100.0))
            start = end
            if start == n_training_example:
                start = 0
            end = start + BATCH
            if end > n_training_example:
                end = n_training_example

        test_accuracy = sess.run(evaulation_step, feed_dict={
            images: testing_images, labels:testing_labels})
        print('Final test accuracy = %.1f%%' % (test_accuracy * 100))

if __name__ == '__main__':
    main()
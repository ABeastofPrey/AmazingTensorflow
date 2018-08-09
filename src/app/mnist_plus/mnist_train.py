import os
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import mnist_inference

# 配置神经网络的参数
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARATION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99
# 模型的存储路径和文件名
MODEL_PATH = "tmp"
MODEL_NAME = "model.ckpt"

def train(mnist):
    # 输入输出的placeholder
    # x = tf.placeholder(name='x-input', shape=[None, mnist_inference.INPUT_NODE], dtype=tf.float32)
    x = tf.placeholder(name='x-input', shape=[BATCH_SIZE, mnist_inference.IMAGE_SIZE, mnist_inference.IMAGE_SIZE, mnist_inference.NUM_CHANNELS], dtype=tf.float32)
    y_ = tf.placeholder(name='y-input', shape=[None, mnist_inference.OUTPUT_NODE], dtype=tf.float32)
    
    regularizer = tf.contrib.layers.l2_regularizer(REGULARATION_RATE)
    # # 直接用mnist_inference中定义的前向传播过程
    y = mnist_inference.inference(x, True, regularizer)

    # 定义存储训练论数的变量。
    # 这个变量不需要计算滑动平均值，所以这里指定这个变量为不可训练的变量（trainable=False）。
    # 在使用tensorflow训练神经网络时，一般会将代表训练轮数的参数指定为不可训练的参数。
    global_step = tf.Variable(0, trainable=False)

    ## 定义损失函数，学习率，滑动平均操作以及训练过程

    # 给定滑动平均衰减率和训练轮数的变量，初始化滑动平均类。
    # 给定训练轮输的变量会加快训练早期变量的更新速度。
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)

    # 在所有代表神经网络参数的变量上使用滑动平均。
    # 其他的辅助变量（比如global_step）就不要了。
    # tf.trainable_variables返回的就是图上集合GraphKeys.TRAINABLE_VARIABLE中的元素。
    # 这个集合的元素就是所有没有指定trainable=False的参数。
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    # 计算交叉商作为刻画预测值和真是值之间的损失函数。
    # 当分类问题只有一个正确答案时，使用sparse_softmax_cross_entropy_with_logits来加速交叉商的计算。
    # 这个函数的第一个参数是神经网络不包括softmax层的前向传播结果，第二个是训练数据的正确答案。
    # 因为标准答案是一个长度为10的一维数组，而该函数需要提供的是一个正确答案的数字，
    # 所以需要使用tf.argmax函数来得到正确答案对应的类别编号。
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))

    # 计算当前batch中所有样列的交叉商平均值。
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    # 总损失等于交叉商损失和正则化损失的和。
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))

    ## 使用指数衰减法控制参数的更新速度。
    # 设置指数衰减的学习率。
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,     # 基础学习率，随着迭代的进行，更新变量的时使用的学习率在这个基础上递减。
        global_step,            # 当前迭代的轮数。
        mnist.train.num_examples / BATCH_SIZE,  # 过完所有的训练需要的迭代次数。
        LEARNING_RATE_DECAY     # 学习率的衰减速度。
    )

    # 使用tf.train.GradientDescentOptimizer优化算法来优化损失函数。
    # 注意这里损失函数包含了交叉商损失和L2正则化损失。
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    
    # 在训练神经网络模型时，每过一遍数据既需要通过反向传播来更新神经网络中的参数，又需要跟新每一个参数的滑动平均值。
    # 为了一次完成多个操作，Tensorflow提供了tf.control_dependencies和tf.group两种机制。
    # 下面两行程序和train_op = tf.group(train_step, variable_averages_op)是等价的。
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name='train')

    # 初始化tensorflow的持久化类
    saver = tf.train.Saver()

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        # 在训练过程中不再测试模型在验证数据上的表现，验证和测试将会有一个独立的程序来完成
        for i in range(TRAINING_STEPS):
            # 产生这一轮使用的一个batch的训练数据，并运行训练过程。
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            reshaped_xs = np.reshape(xs, (BATCH_SIZE, mnist_inference.IMAGE_SIZE, mnist_inference.IMAGE_SIZE, mnist_inference.NUM_CHANNELS))
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={ x: reshaped_xs, y_: ys })
            
            # 每1000轮保存一次模型
            if i % 1000 == 0:
                # 输出当前的训练结果。
                # 这里只输出模型在当前训练batch上的损失函数的大小，通过损失函数的大小可以大概了解到模型的训练情况。
                # 在验证数据集上的信息将会有一个独立的程序来完成。
                print("After %d training step(s), loss on training batch is %g."%(step, loss_value))

                # 保存当前的模型。
                # 这里给出了global_step参数，这样可以让每个被保存模型的文件名末尾加上训练的轮数。
                # 比如"model.ckpt-1000"表示1000轮训练之后得到的模型。
                saver.save(sess, os.path.join(MODEL_PATH, MODEL_NAME), global_step=global_step)

def printer(mnist):
    # Training data size.
    print('Training data size: ', mnist.train.num_examples)
    # Validation data size.
    print('Validation data size: ', mnist.validation.num_examples)
    # Example training data.
    # print('Example training data: \n', mnist.train.images[0])
    print('Example training data length: \n', len(mnist.train.images[0]))
    # Example training data label.
    print('Example training data label: \n', mnist.train.labels[0])

def main(argv=None):
    # 声明处理MNIST数据集的类，这个类会在初始化时自动下载数据
    mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)
    # printer(mnist)
    train(mnist)

    # tensorflow提供的一个主程序入口，tf.app.run会调用上面定义的main函数
if __name__ == '__main__':
    tf.app.run()
import tensorflow as tf
from mnist import input_data

# 配置神经网络的参数
INPUT_NODE = 784                # 输入层节点数，每张图片是28*28像素，所以 INPUT_NODE = 784。
OUTPUT_NODE = 10                # 输出层是0～9的数字，所以 OUTPUT_NODE = 10。
LAYER1_NODE = 500               # 第一个隐含层设置500个节点。
BATCH_SIZE = 100                # 一个训练batch中训练数据的个数。数字越小，训练过程越接近随机梯度下降；数字越大，训练过程越接近梯度下降。
LEARNING_RATE_BASE = 0.8        # 基础学习率
LEARNING_RATE_DECAY = 0.99      # 学习率的衰减率
REGULARIZATION_RATE = 0.0001    # 描述模型复杂度的正则化项在损失函数中的系数
TRAINING_STEPS = 30000          # 训练次数
MOVING_AVERAGE_DECAY = 0.99     # 滑动平均衰减率

# 一个辅助函数，给定神经网络的输入和所有参数，计算神经网络的前向传播结果。
# 在这里定义了一个使用ReLU激活层的三个全联接神经网络。
# 通过加入了一个隐含层实现了多层网络结构，通过ReLU实现了去线性化。
# 在这个函数中也支持传入用于计算参数平均值的类，这样方便在测试时使用滑动平均模型。
def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):
    # 当没提供滑动平均类时，直接使用参数当前的取值
    if avg_class == None:
        # 计算隐含层的前向传播结果，这里使用了ReLU激活函数
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)

        # 计算输出层的前向传播结果。因为在计算损失函数时会一并计算softmax函数，所以这里不需要加入激活函数。
        # 而且不加入softmax不影响计算结果。
        # 因为预测时使用的是不同类别对应节点输出值的相对大小，有没有softmax层对最后分类结果的计算没有影响。
        # 于是在计算整个神经网络的前向传播时可以不加入最后的softmax层。
        return tf.matmul(layer1, weights2) + biases2
    else:
        # 首先使用avg_class.average函数来计算得出变量的滑动平均值，
        # 然后再计算计算相应的神经网络的前向传播结果。
        layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights1)) + avg_class.average(biases1))
        return tf.matmul(layer1, avg_class.average(weights2)) + avg_class.average(biases2)

# 模型的训练过程
def train(mnist):
    x = tf.placeholder(name='x-input', shape=[None, INPUT_NODE], dtype=tf.float32)
    y_ = tf.placeholder(name='y-input', shape=[None, OUTPUT_NODE], dtype=tf.float32)

    # 生成隐含层的参数
    weights1 = tf.Variable(tf.truncated_normal(shape=[INPUT_NODE, LAYER1_NODE], stddev=0.1))
    biases1 = tf.Variable(tf.constant(value=0.1, shape=[LAYER1_NODE]))

    # 生成输出层的参数
    weights2 = tf.Variable(tf.truncated_normal(shape=[LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
    biases2 = tf.Variable(tf.constant(shape=[OUTPUT_NODE], value=0.1))

    # 计算当前参数下神经网络的前向传播结果。这里给出的计算滑动平均的类为None.
    # 所以函数不会使用参数的华东平均值。
    y = inference(x, None, weights1, biases1, weights2, biases2)

    # 定义存储训练论数的变量。
    # 这个变量不需要计算滑动平均值，所以这里指定这个变量为不可训练的变量（trainable=False）。
    # 在使用tensorflow训练神经网络时，一般会将代表训练轮数的参数指定为不可训练的参数。
    global_step = tf.Variable(0, trainable=False)

    # 给定滑动平均衰减率和训练轮数的变量，初始化滑动平均类。
    # 给定训练轮输的变量会加快训练早期变量的更新速度。
    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)

    # 在所有代表神经网络参数的变量上使用滑动平均。
    # 其他的辅助变量（比如global_step）就不要了。
    # tf.trainable_variables返回的就是图上集合GraphKeys.TRAINABLE_VARIABLE中的元素。
    # 这个集合的元素就是所有没有指定trainable=False的参数。
    ema_op = ema.apply(tf.trainable_variables())

    # 计算使用了滑动平均之后的前向传播结果。
    # 滑动平均不会改变变量本身的取值，而是维护一个影子变量来记录其滑动平均值。
    # 所以当要使用这个滑动平均值时，需要明确调用average函数。
    average_y = inference(x, ema, weights1, biases1, weights2, biases2)
    
    # 计算交叉商作为刻画预测值和真是值之间的损失函数。
    # 当分类问题只有一个正确答案时，使用sparse_softmax_cross_entropy_with_logits来加速交叉商的计算。
    # 这个函数的第一个参数是神经网络不包括softmax层的前向传播结果，第二个是训练数据的正确答案。
    # 因为标准答案是一个长度为10的一维数组，而该函数需要提供的是一个正确答案的数字，
    # 所以需要使用tf.argmax函数来得到正确答案对应的类别编号。
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))

    # 计算当前batch中所有样列的交叉商平均值。
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    # 计算L2正则化损失函数。
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)

    # 计算模型的正则化损失。
    # 一般只计算网络边上权重的正则化损失，而不计算偏置项。
    regularization = regularizer(weights1) + regularizer(weights2)

    # 总损失等于交叉商损失和正则化损失的和。
    loss = cross_entropy_mean + regularization
    
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
    # 下面两行程序和train_op = tf.group(train_step, ema_op)是等价的。
    with tf.control_dependencies([train_step, ema_op]):
        train_op = tf.no_op(name='train')

    # 检验使用了滑动平均模型的神经网络前向传播结果是否正确。
    # tf.argmax(average_y, 1)计算每一个样列的预测答案。其中average_y是一个batch_size * 10的二维数组，
    # 每一行表示一个样列的前向传播结果。
    # tf.argmax的第二个参数“1”表示选取最大值的操作仅在第一个纬度中进行，也就是说，
    # 只在每一行选取最大值对应的下标。
    # 于是得到结果是一个长度为batch的一维数组，这个一维数组中的值就表示了每一个样列对应的数字识别的结果。
    # tf.equal判断两个张量的每一维是否相等，是返回True，否则返回False。
    correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))

    # 这个运算首先将一个布尔型数值转换成实数型，然后计算平均值。
    # 这个平均值就是模型在这组数据上的正确率。
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 初始化会话并开始训练。
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        # 准备验证数据。
        # 一般在神经网络的训练过程中会通过验证数据来大致判断停止的条件和评判训练的效果。
        validate_feed = {
            x: mnist.validation.images,
            y_: mnist.validation.labels
        }

        # 准备测试数据。
        # 在真是的应用中，这部分数据在训练时是不可见的，这个数据只是作为模型优劣的最后评价标准。
        test_feed = {
            x: mnist.test.images,
            y_: mnist.test.labels
        }

        # 迭代训练神经网络。
        for i in range(TRAINING_STEPS):
            # 每1000轮输出一次在验证数据上的测试结果。
            if i % 1000 == 0:
                # 计算滑动平均模型在验证数据上的结果。
                # 因为MNIST数据集比较小，所以一次可以处理所有的数据。
                # 为了计算方便，本样列程序没有将验证数据划分为更小的batch。
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)

                # 计算滑动平均模型在测试数据上的结果。
                test_acc = sess.run(accuracy, feed_dict=test_feed)
                print('After %d training step(s), validation accuracy using average model is %g, test accuracy using average model is %g' % (i, validate_acc, test_acc))
            
            # 产生这一轮使用的一个batch的训练数据，并运行训练过程。
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op, feed_dict={ x: xs, y_: ys })

        # 在训练结束之后，在测试数据上检测神经网络模型的最终正确率。
        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print('After %d training step(s), test accuracy using average model is %g' % (i, validate_acc))        
    # pass

# 主程序入口
def main(argv=None):
    # 声明处理MNIST数据集的类，这个类会在初始化时自动下载数据
    mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)
    train(mnist)
    # pass

# tensorflow提供的一个主程序入口，tf.app.run会调用上面定义的main函数
if __name__ == '__main__':
    tf.app.run()
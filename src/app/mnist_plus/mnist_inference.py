import tensorflow as tf

# 定义神经网络相关参数
INPUT_NODE = 784                # 输入层节点数，每张图片是28*28像素，所以 INPUT_NODE = 784。
OUTPUT_NODE = 10                # 输出层是0～9的数字，所以 OUTPUT_NODE = 10。

IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10

# 第一层卷积层的尺寸和深度。
CONV1_DEEP = 32
CONV1_SIZE = 5

# 第二层卷积层的尺寸和深度。
CONV2_DEEP = 64
CONV2_SIZE = 5

# 全连接节点的个数
FC_SIZE = 512

# 定义神经网络的前向传播过程。
# train参数用于区分训练过程和测试过程。
# dropout用于进一步提升模型的可靠性，并防止过拟合。只在训练过程使用。
def inference(input_tensor, train, regularizer):
    # 申明第一层卷积层的变量，并是实现前向传播。
    # 输入为28*28*1的原始MNIST像素，因为使用了全0填充，所以输出为28*28*32的矩阵。
    with tf.variable_scope('layer1_conv1'):
        conv1_weights = tf.get_variable(name='wight', shape=[CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP], initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable(name='bias', shape=[CONV1_DEEP], initializer=tf.constant_initializer(0.0))
        # 用边长为5，深度为32的过滤器，移动步长为1，使用0填充。
        conv1 = tf.nn.conv2d(input=input_tensor, filter=conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))
    
    # 第二层池化层的前向传播过程。
    # 选择最大池化层，过滤器边长为2，步长为2，全0填充。
    # 输入为上一层的输出，28*28*32的矩阵，输出为14*14*32的矩阵。
    with tf.name_scope('layer2_pool1'):
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    

    # 申明第三层卷积层的变量，并是实现前向传播。
    # 输入为14*14*32的矩阵，输出为14*14*64的矩阵。
    with tf.variable_scope('layer3_conv2'):
        conv2_weights = tf.get_variable(name='weights', shape=[CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP], initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable(name='bias', shape=[CONV2_DEEP], initializer=tf.constant_initializer(0.0))
        # 使用边长为5，深度为64的过滤器，步长为1，使用0填充。
        conv2 = tf.nn.conv2d(input=pool1, filter=conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
        relue2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    # 实现第四层池化层的前向传播过程。
    # 这一层和第二层的结构是一样的。
    # 输入为14*14*64的矩阵，输出为7*7*64的矩阵。
    with tf.name_scope('layer4_pool2'):
        pool2 = tf.nn.max_pool(relue2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # 将第四层的输出转化成第五层全连接层的输入格式。
    # 第四层的输出为7*7*64的矩阵，可第五层全连接层需要的格式为向量，所以需要将7*7*64的矩阵拉直成一维向量。
    # 注意：因为每一层神经网络的输入输出都为一个batch的矩阵，所以这里得到的矩阵也包含一个batch中数据的个数。
    pool_shape = pool2.get_shape().as_list()
    # 计算将矩阵拉直成向量后的长度，这个长度就是矩阵长宽及深度的乘积。
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
    # 通过tf.reshape将第四层的输出变成一个batch的数量。
    reshaped = tf.reshape(pool2, [pool_shape[0], nodes])

    # 申明第五层全连接层的变量，并是实现前向传播。
    # 这一层的输入是拉直后长度为3136的向量，输出是一组长度为120的向量。
    # 这里使用了dropout，使得在训练时会随机将部分节点的数据改为0，从而避免过拟合问题，使得模型在测试数据上的效果更好。
    # dropout一般只在全连接层而不是在卷积层或者池化层使用。
    with tf.variable_scope('layer5_fc1'):
        fc1_weights = tf.get_variable('weight', shape=[nodes, FC_SIZE], initializer=tf.truncated_normal_initializer(stddev=0.1))
        # 只在全连接层加入正则化。
        if not regularizer is None:
            tf.add_to_collection('losses', regularizer(fc1_weights))
        fc1_biases = tf.get_variable('bias', shape=[FC_SIZE], initializer=tf.constant_initializer(0.0))
        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
        if train:
            fc1 = tf.nn.dropout(fc1, 0.5)
    
    # 申明第五层全连接层的变量，并是实现前向传播。
    # 这一层的输入是120的向量，输出是一组长度为10的向量。
    # 最后通过softmax得到分类结果的置信度。
    with tf.variable_scope('layer6_fc2'):
        fc2_weights = tf.get_variable(name='weight', shape=[FC_SIZE, NUM_LABELS], initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fc2_weights))
        fc2_biases = tf.get_variable(name='bias', shape=[NUM_LABELS], initializer=tf.constant_initializer(0.1))
        logit = tf.matmul(fc1, fc2_weights) + fc2_biases
        return logit

# Initial mnist data and import and test it.
import tensorflow as tf
from mnist import input_data

# Imports training data
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

def printer():
    # Training data size.
    print('Training data size: ', mnist.train.num_examples)
    # Validation data size.
    print('Validation data size: ', mnist.validation.num_examples)
    # Example training data.
    # print('Example training data: \n', mnist.train.images[0])
    print('Example training data length: \n', len(mnist.train.images[0]))
    # Example training data label.
    print('Example training data label: \n', mnist.train.labels[0])

# 从所有训练数据中去除以下部分进行训练的batch功能
def batchFn():
    batch_size = 100
    xs, ys = mnist.train.next_batch(batch_size)
    print('Xs shape: ', xs.shape) # training data
    print('Ys shape: ', ys.shape) # training data label

# 主程序入口，只是为了测试数据集是否加载成功。
def main(argv=None):
    # 打印测试结果。
    printer()
    # 测试batch功能。
    batchFn()

if __name__ == '__main__':
    tf.app.run()    # tensorflow提供的一个主程序入口，tf.app.run会调用上面定义的main函数
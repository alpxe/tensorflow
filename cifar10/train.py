import tensorflow as tf
import cv2
import numpy as np


def _analyze(buffer):
    bytes = tf.decode_raw(buffer, tf.uint8)  # 字符字节转int

    label = tf.cast(tf.strided_slice(bytes, [0], [1]), dtype=tf.int32)
    label = tf.reshape(label, [])

    label = tf.one_hot(label, 10)

    image = tf.strided_slice(bytes, [1], [32 * 32 * 3 + 1])
    image = tf.reshape(image, [3, 32, 32])
    # Convert from [depth, height, width] to [height, width, depth].
    image = tf.transpose(image, [1, 2, 0])
    return label, image  # BGR


def weight_var(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_var(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):  # 卷积核移动步长1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")


def pool_max(x):  # 池化大小2*2，池化步长2，池化类型为最大池化
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    pass


# 文件路径
filenames = ["cifar10_data/cifar-10-batches-bin/data_batch_{0}.bin".format(i) for i in range(1, 6)]
dataset = tf.data.FixedLengthRecordDataset(filenames, 1 + 32 * 32 * 3)

dataset = dataset.repeat()  # 此数据集会循环

# 解析一组 1+32*32*3 的数据
dataset = dataset.map(_analyze)

dataset = dataset.batch(2)

iterator = dataset.make_one_shot_iterator()
label, image = iterator.get_next()

image = tf.cast(image, dtype=tf.uint8, name="imgX")
image = tf.cast(image / 0xff, tf.float32)  # 化作浮点数  [N,32,32,3]

# 第一层卷积
with tf.name_scope("conv1"):
    kernel = weight_var([3, 3, 3, 64])  # 卷积核 大小3*3 深度3 卷积核个数64
    bias = bias_var([64])  # 置顶项
    conv = tf.nn.relu(conv2d(image, kernel) + bias)  # 激活+卷积
    pool = pool_max(conv)

# 第二层卷积
with tf.name_scope("conv2"):
    kernel = weight_var([3, 3, 64, 128])
    bias = bias_var([128])
    conv = tf.nn.relu(conv2d(pool, kernel) + bias)
    pool = pool_max(conv)
    pass

# 第三层卷积
with tf.name_scope("conv3"):
    kernel = weight_var([3, 3, 128, 256])
    bias=bias_var([256])
    conv=tf.nn.relu(conv2d(pool,kernel)+bias)
    pool=pool_max(conv)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    p=sess.run(pool)
    print(p.shape)
    # a=sess.run(bias_var([2,2,2,3]))
    # print(a)
    # # print(sess.run(image))
    #
    # # img = sess.run(iterator.get_next())
    # #
    # # img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    # # cv2.imshow('w', img)
    # # cv2.waitKey()
    pass

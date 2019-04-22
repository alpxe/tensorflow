import tensorflow as tf
import cv2
import numpy as np


# W权重变量
def weight_var(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# b偏置项变量
def bias_var(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# 卷积
def conv_2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")


def resnet(x):
    pass


# 解析一组 1+32*32*3的数据
def _analyze(buffer):
    bytes = tf.decode_raw(buffer, tf.uint8)

    label = tf.cast(tf.strided_slice(bytes, [0], [1]), dtype=tf.int32)
    label = tf.reshape(label, shape=[])
    label = tf.one_hot(label, 10)

    image = tf.strided_slice(bytes, [1], [1 + 32 * 32 * 3])
    image = image / 0xFF

    image = tf.reshape(image, [3, 32, 32])
    image = tf.transpose(image, [1, 2, 0])

    return label, image
    pass


filenames = ["cifar10_data/cifar-10-batches-bin/data_batch_{0}.bin".format(i) for i in range(1, 6)]
dataset = tf.data.FixedLengthRecordDataset(filenames, 1 + 32 * 32 * 3)

dataset = dataset.repeat()  # 数据循环
dataset = dataset.map(_analyze)
dataset = dataset.batch(1)

iter = dataset.make_one_shot_iterator()

label, image = iter.get_next()

res=image

v1_W = weight_var([2, 2, 3, 32])
v1_conv=conv_2d(image,v1_W)

res+=v1_conv

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())


    print(sess.run(res))
    # img = sess.run(image)
    # img = img[0]
    # print(img)
    # img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    # cv2.imshow("w", img)
    # cv2.waitKey()

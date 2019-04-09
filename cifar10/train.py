import tensorflow as tf
import cv2
import numpy as np
import os
import shutil


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

dataset = dataset.batch(50)

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
    bias = bias_var([256])
    conv = tf.nn.relu(conv2d(pool, kernel) + bias)
    pool = pool_max(conv)

# 全层连接
with tf.name_scope("conn"):
    full = tf.reshape(pool, [-1, 4 * 4 * 256])
    connW = weight_var([4 * 4 * 256, 1024])
    connB = weight_var([1024])
    layer = tf.nn.relu(tf.matmul(full, connW) + connB)

    prob = tf.placeholder(dtype=tf.float32, name="prob")  # 训练时 传入0.5 50%概率drop  使用时传入1
    layer_drop = tf.nn.dropout(layer, keep_prob=prob)

with tf.name_scope("logit"):
    ltW = weight_var([1024, 10])
    ltB = bias_var([10])

    logit = tf.add(tf.matmul(layer_drop, ltW), ltB)
    softmax = tf.nn.softmax(logit)

    softmax = tf.clip_by_value(softmax, 1e-10, 1.0)  # 值保护不会出现0和大于1

label = tf.cast(label, dtype=tf.float32)

# 交叉熵构造损失              H(x)=-∑P(xᵢ)·log₂[P(xᵢ)]
loss = tf.reduce_mean(-tf.reduce_sum(label * (tf.log(softmax) / tf.log(2.)), 1))
tf.summary.scalar("loss", loss)

# loss=tf.nn.softmax_cross_entropy_with_logits_v2(labels=label,logits=logit)

train = tf.train.AdamOptimizer(1e-5).minimize(loss)

# 定义测试的准确率
correct_prediction = tf.equal(tf.argmax(softmax, 1), tf.argmax(label, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar("accuracy", accuracy)

###
def saved_model(sess):
    dir_path = "model_data/"

    if os.path.exists(dir_path):
        print("文件夹已存在，先执行删除")
        shutil.rmtree(dir_path)

    # 保存
    builder = tf.saved_model.builder.SavedModelBuilder(dir_path)
    builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.TRAINING])
    builder.save()


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    merge = tf.summary.merge_all()
    writer = tf.summary.FileWriter("graph/", graph=sess.graph)

    for i in range(10000):

        _, ls = sess.run([train, loss], feed_dict={prob: 0.5})

        if i % 50 == 0:
            _, ls,acc, m = sess.run([train, loss,accuracy, merge], feed_dict={prob: 0.5})

            writer.add_summary(m, i)
            print("step:{0}  loss:{1}  acc:{2}".format(i, ls,acc))

        if i % 1000 == 0:
            saved_model(sess)

    print("train complete！！！")
    saved_model(sess)

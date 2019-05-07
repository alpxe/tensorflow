import tensorflow as tf
from tensorflow.python.training import moving_averages
import numpy as np


def _resolve(buf):
    fats = tf.parse_single_example(buf, features={
        "label": tf.FixedLenFeature([1], dtype=tf.int64),
        "image": tf.FixedLenFeature([28 * 28], dtype=tf.int64)
    })

    label = tf.reshape(fats["label"], shape=[])
    label = tf.cast(tf.one_hot(label, 10), tf.float32)

    image = tf.cast(tf.reshape(fats["image"] / 0xFF, shape=[28, 28, 1]), tf.float32)

    return label, image


# batch norm layer
def batch_norm(x, decay=0.999, epsilon=1e-03, is_training=True,
               scope="scope"):
    x_shape = x.get_shape()
    num_inputs = x_shape[-1]
    reduce_dims = list(range(len(x_shape) - 1))
    with tf.variable_scope(scope):
        beta = tf.get_variable("bate", shape=[num_inputs, ], initializer=tf.zeros_initializer())
        gamma = tf.get_variable("gamma", shape=[num_inputs, ], initializer=tf.zeros_initializer())

        moving_mean = tf.get_variable("moving_mean", shape=[num_inputs, ], initializer=tf.zeros_initializer(),
                                      trainable=False)
        moving_variance = tf.get_variable("moving_variance", shape=[num_inputs, ], initializer=tf.zeros_initializer(),
                                          trainable=False)

    if is_training:
        mean, variance = tf.nn.moments(x, axes=reduce_dims)
        update_move_mean = moving_averages.assign_moving_average(moving_mean,
                                                                 mean, decay=decay)
        update_move_variance = moving_averages.assign_moving_average(moving_variance,
                                                                     variance, decay=decay)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_move_mean)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_move_variance)
    else:
        mean, variance = moving_mean, moving_variance
    return tf.nn.batch_normalization(x, mean, variance, beta, gamma, epsilon)


def conv2d(x, out, kernel_size, stride, scope):
    """
    卷积
    :param x: 输入
    :param out: 输出的维度
    :param kernel_size: 卷积核尺寸
    :param stride: 卷积核滑动
    :param scope:
    :return:
    """

    align = x.get_shape()[-1]  # 获取输入最后的维度 用于维度对齐
    with tf.variable_scope(scope):
        # 创建卷积核 [ 尺寸 , 尺寸 , 与输入最后的维度一致 , 输出的维度 ]
        kernel = tf.get_variable("kernel",
                                 shape=[kernel_size, kernel_size, align, out],
                                 dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())

        # 卷积方法
        return tf.nn.conv2d(x, kernel, strides=[1, stride, stride, 1], padding="SAME")
    pass


def max_pool(x, pool_size, stride, scope):
    with tf.variable_scope(scope):
        return tf.nn.max_pool(x, [1, pool_size, pool_size, 1],
                              [1, stride, stride, 1], padding="SAME")


def avg_pool(x, pool_size, scope):
    with tf.variable_scope(scope):
        return tf.nn.avg_pool(x, [1, pool_size, pool_size, 1],
                              strides=[1, pool_size, pool_size, 1], padding="VALID")


def _bottleneck(x, d, out, stride=None, scope="bottleneck"):
    """
    三层瓶颈结构为1X1，3X3和1X1卷积层
    其中两个1X1卷积用来减少或增加维度
    3X3卷积可以看作一个更小的输入输出维度的瓶颈
    :param x: 输入
    :param d: 瓶颈值 教程上是out整除4的值
    :param out: 输出的维度值
    :return:
    """

    align = x.get_shape()[-1]  # 获取输入最后的维度 用于维度对齐

    if stride is None:
        stride = 1 if align == out else 2

    with tf.variable_scope(scope):
        # 1x1卷积核 如果stride默认且输入与输出的维度一致，则步长为2 卷积后的size/2
        h = conv2d(x, d, 1, stride=stride, scope="conv_1")  # [batch,size,size,align]->[batch,size,size,d]
        h = batch_norm(h, scope="bn_1")
        h = tf.nn.relu(h)

        # 3x3卷积核 [batch,size,size,d]->[batch,size,size,d]
        h = conv2d(h, d, 3, stride=1, scope="conv_2")
        h = batch_norm(h, scope="bn_2")
        h = tf.nn.relu(h)

        # 1x1卷积核 [batch,size,size,d]->[batch,size,size,out]
        h = conv2d(h, out, 1, stride=1, scope="conv_3")
        h = batch_norm(h, scope="bn_3")

        if align != out:  # 维度不同
            shortcut = conv2d(x, out, 1, stride=stride, scope="conv_4")
            shortcut = batch_norm(shortcut, scope="bn_4")
        else:
            shortcut = x

        return tf.nn.relu(h + shortcut)
    pass


def _block(x, out, n, init_stride=2, scope="block"):
    """
    残差
    :param x: 输入
    :param out: 输出的维度
    :param n: 迭代次数
    :param init_stride: 初始步长值
    :param scope:
    :return:
    """

    with tf.variable_scope(scope):
        bok = out // 4  # 瓶颈值
        net = _bottleneck(x, bok, out, stride=init_stride, scope="bottlencek1")

        for i in range(1, n):
            net = _bottleneck(net, bok, out, scope=("bottlencek%s" % (i + 1)))

        return net
    pass


data = tf.data.TFRecordDataset("tfrecords/train.tfrecords")
data = data.repeat()
data = data.map(_resolve)
data = data.batch(5)

iterator = data.make_one_shot_iterator()
label, image = iterator.get_next()

with tf.variable_scope("input"):
    image = tf.cast(image, tf.float32, name="image")

# 残差神经网络
with tf.variable_scope("resnet"):
    print("net    -- shape -- :")

    net = conv2d(image, 32, 3, 1, scope="conv1")  # [10,28,28,1] -> [10,28,28,32]
    net = tf.nn.relu(batch_norm(net, scope="bn1"))
    net = max_pool(net, 2, 2, "maxpool1")  # ->[10,14,14,32]
    tf.summary.histogram("resnet/conv1", net)

    net = _block(net, 256, 3, 1, scope="block_2")  # [?,14,14,256]
    tf.summary.histogram("resnet/conv2", net)

    net = _block(net, 512, 4, 1, scope="block3")  # [?,14,14,512]
    tf.summary.histogram("resnet/conv3", net)

    net = _block(net, 1024, 3, scope="block4")  # [?,7,7,1024]
    print(net.get_shape())
    tf.summary.histogram("resnet/conv4", net)

    # net = _block(net, 2048, 3, scope="block5")  # [?,2,2,2048]
    # tf.summary.histogram("resnet/conv5", net)

    net = avg_pool(net, 7, scope="avgpool5")  # [?,1,1,1024]
    print(net.get_shape())

    net = tf.squeeze(net, [1, 2], name="SpatialSqueeze")  # -> [batch, 2048]
    print(net.get_shape())

    print(" --- --- ---\n")

with tf.variable_scope("logit"):
    align = net.get_shape()[-1]

    weight = tf.get_variable("weight", shape=[align, 10], dtype=tf.float32,
                             initializer=tf.contrib.layers.xavier_initializer())
    bias = tf.get_variable("bias", shape=[10], dtype=tf.float32, initializer=tf.zeros_initializer())

    logit = tf.nn.xw_plus_b(net, weight, bias)
    softmax = tf.clip_by_value(tf.nn.softmax(logit), 1e-10, 1.0, name="softmax")  # 值保护不会出现0和大于1

    loss = tf.reduce_mean(-tf.reduce_sum(label * (tf.log(softmax) / tf.log(2.)), 1))
    tf.summary.scalar('loss', loss)  # 与tensorboard 有关

    train = tf.train.AdamOptimizer(0.01).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    merge = tf.summary.merge_all()  # tensorboard
    writer = tf.summary.FileWriter("graph/", graph=sess.graph)  # 写入日志log

    for i in range(500 + 1):
        sess.run([train, loss])

        if i % 10 == 0:
            _, ls, mrg, sx = sess.run([train, loss, merge, softmax])
            writer.add_summary(mrg, i)

            print("loss损失值:%s" % ls)
            pass
        pass

    writer.close()

    # 训练完毕 保存模型文件
    builder = tf.saved_model.builder.SavedModelBuilder("model_data/")
    builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.TRAINING])
    builder.save()
    pass

import tensorflow as tf


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):  # 卷积核移动步长1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")


def pool_max_2x2(x):  # 池化大小2*2，池化步长2，池化类型为最大池化
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    pass


def resnet(x, filter):
    res = x


    pass


def __format(record):
    fats = tf.parse_single_example(record, features={
        "label": tf.FixedLenFeature([1], dtype=tf.int64),
        'image': tf.FixedLenFeature([28 * 28], dtype=tf.int64)
    })

    label = tf.reshape(fats["label"], [])
    label = tf.one_hot(label, depth=10)

    image = tf.reshape(fats["image"] / 255, shape=[28, 28, 1])
    # image = tf.cast(image, tf.float32)
    return label, image


# 读取数据集文件
dataset = tf.data.TFRecordDataset("MNIST_data/test.tfrecords")
dataset = dataset.repeat()  # 重复此数据集

dataset = dataset.map(__format)  # 数据格式化

dataset = dataset.batch(100)  # 批次量 每次输出的个数

iterator = dataset.make_one_shot_iterator()
label, image = iterator.get_next()  # 获取

label = tf.cast(label, tf.float32)
image = tf.cast(image, tf.float32)

# 这里 image的shape=[?,28,28,1] 通过卷积-> [?,28,28,32]
conv_k1 = weight_variable([3, 3, 1, 32])  # 卷积核
conv_b1 = bias_variable([32])

conv_v1 = tf.nn.relu(conv2d(image, conv_k1) + conv_b1)  # 第一次卷积

# 池化  [?,28,28,32] -> [?,14,14,32]
pool_v1 = pool_max_2x2(conv_v1)
print(pool_v1.shape)

# 第二次卷积 [?,14,14,32]->[?,14,14,64]
conv_k2 = weight_variable([3, 3, 32, 64])
conv_b2 = bias_variable([64])

conv_v2 = tf.nn.relu(conv2d(pool_v1, conv_k2) + conv_b2)

# 池化 [?,14,14,64]-->[?,7,7,64]
pool_v2 = pool_max_2x2(conv_v2)

# 卷积 得到10个特征
conv_k = weight_variable([3, 3, 64, 10])
conv_b = bias_variable([10])
conv_v = tf.nn.relu(conv2d(pool_v2, conv_k) + conv_b)  # [?,7,7,10]

# 全局池化 avg_pool
avg = tf.nn.avg_pool(conv_v, ksize=[1, 7, 7, 1], strides=[1, 7, 7, 1], padding="SAME")
# [?,1,1,10]

logit = tf.reshape(avg, [-1, 10])

softmax = tf.nn.softmax(logit)
softmax = tf.clip_by_value(softmax, 1e-10, 1.0)  # 值保护不会出现0和大于1

# 交叉熵构造损失              H(x)=-∑P(xᵢ)·log₂[P(xᵢ)]
loss = tf.reduce_mean(-tf.reduce_sum(label * (tf.log(softmax) / tf.log(2.)), 1))
train = tf.train.AdamOptimizer(1e-4).minimize(loss)

# 训练集上的准确率
with tf.name_scope("accuracy"):
    # 定义测试的准确率
    correct_prediction = tf.equal(tf.argmax(softmax, 1), tf.argmax(label, 1))
    print("----------------")
    print(softmax.shape)
    print(label.shape)
    print("---------------")
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(1000):
        _, ls = sess.run([train, loss])
        print("loss:{0}".format(ls))

        if i % 50 == 0:
            _, acc = sess.run([train, accuracy])
            print("step:{0} 准确率：{1:.5}".format(i, acc))
    # print(sess.run(softmax))

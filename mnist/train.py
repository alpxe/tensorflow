"""
深度学习
"""
import tensorflow as tf


def __format(record):
    fats = tf.parse_single_example(record, features={
        "label": tf.FixedLenFeature([1], dtype=tf.int64),
        'image': tf.FixedLenFeature([28 * 28], dtype=tf.int64)
    })

    label = tf.one_hot(fats["label"], depth=10)
    image = tf.reshape(fats["image"] / 255, shape=[28, 28, 1])
    # image = tf.cast(image, tf.float32)
    return label, image


# 读取数据集文件
dataset = tf.data.TFRecordDataset("tfrecords/test.tfrecords")
dataset = dataset.repeat()  # 重复此数据集

dataset = dataset.map(__format)  # 数据格式化

dataset = dataset.batch(100)  # 批次量 每次输出的个数

iterator = dataset.make_one_shot_iterator()
label, image = iterator.get_next()  # 获取

def delimit_W(shape):
    initial = tf.truncated_normal(shape, stddev=1e-5)
    return tf.Variable(initial_value=initial)


def delimit_B(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial_value=initial)


# 卷积
def conv(input, kernel):
    return tf.nn.relu(tf.nn.conv2d(input, kernel, strides=[1, 1, 1, 1], padding="SAME"))


def pool(value):
    # 第一个参数value：需要池化的输入，一般池化层接在卷积层后面，所以输入通常是feature map，依然是[batch, height, width, channels]这样的shape
    # 第二个参数ksize：池化窗口的大小，取一个四维向量，一般是[1, height, width, 1]，因为我们不想在batch和channels上做池化，所以这两个维度设为了1
    # 第三个参数strides：和卷积类似，窗口在每一个维度上滑动的步长，一般也是[1, stride, stride, 1]
    # 第四个参数padding：和卷积类似，可以取'VALID' 或者'SAME'
    # return 返回一个Tensor，类型不变，shape仍然是[batch, height, width, channels]这种形式。  这里height.width /2
    return tf.nn.max_pool(value, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


with tf.name_scope("conv1"):
    # 设置一个卷积核参数 3x3大小  根据images的通道：[1]  ->生成[32]个维度
    W_conv1 = delimit_W([5, 5, 1, 32])

    # 卷积   shepe=（将image[-1,28,28,1] 通过[3,3,1,32] ->[-1,28,28,32]）
    img=tf.cast(image,tf.float32,name="img")
    out_conv1 = conv(img, W_conv1)


    # 池化 #[-1,28,28,32] -> [-1,14,14,32]
    out_pool1 = pool(out_conv1)

    tf.summary.histogram("conv1/conv", out_conv1)
    tf.summary.histogram("conv1/pool", out_pool1)

with tf.name_scope("conv2"):
    # 卷积参数2
    W_conv2 = delimit_W([3, 3, 32, 64])

    # 第二次卷积 return:[-1,14,14,64]
    out_conv2 = conv(out_pool1, W_conv2)

    # 池化2 return:[-1,7,7,64]
    out_pool2 = pool(out_conv2)

    tf.summary.histogram("conv2/conv", out_conv2)
    tf.summary.histogram("conv2/pool", out_pool2)

with tf.name_scope("fully"):
    # 全层连接
    full_res = tf.reshape(out_pool2, shape=[-1, 7 * 7 * 64])
    full_W = delimit_W([7 * 7 * 64, 1024])
    full_B = delimit_B([1024])
    fly = tf.nn.relu(tf.matmul(full_res, full_W) + full_B)  # return: [-1,1024]

    tf.summary.histogram("fully/fly", fly)

prob = tf.placeholder(dtype=tf.float32, name="prob")  # 训练时 传入0.5 50%概率drop  使用时传入1
fly_drop = tf.nn.dropout(fly, keep_prob=prob)

with tf.name_scope("logit"):
    lgtW = delimit_W([1024, 10])
    lgtB = delimit_B([10])

    logit = tf                    .add(tf.matmul(fly_drop, lgtW), lgtB, name="logit")
    # result=tf.nn.softmax(logit,name="result")
    tf.summary.histogram("logit", logit)

loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(labels=label, logits=logit)
)
tf.summary.scalar('loss', loss)  # 与tensorboard 有关

train = tf.train.AdamOptimizer(1e-4).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    merge = tf.summary.merge_all()  # tensorboard

    writer = tf.summary.FileWriter("graph/", graph=sess.graph)  # 写入日志log

    # saver = tf.train.Saver()  # 初始化 Saver

    for i in range(5000):
        _, ls, mrg = sess.run([train, loss, merge], feed_dict={prob: 0.5})
        writer.add_summary(mrg, i)
        if i % 50 == 0:
            print(ls)
            # saver.save(sess, save_path='ckp/')  # 储存神经网络的变量
        pass
    pass
    writer.close()

    # 训练完毕 保存模型文件
    builder = tf.saved_model.builder.SavedModelBuilder("model_data/")
    builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.TRAINING])
    builder.save()

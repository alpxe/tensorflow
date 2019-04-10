import tensorflow as tf

def __format(record):
    fats = tf.parse_single_example(record, features={
        "label": tf.FixedLenFeature([1], dtype=tf.int64),
        'image': tf.FixedLenFeature([28 * 28], dtype=tf.int64)
    })

    label=tf.reshape(fats["label"],shape=[])
    label = tf.cast(tf.one_hot(label, depth=10),dtype=tf.float32)

    image = tf.reshape(fats["image"] / 0xff, shape=[28, 28, 1])
    # image = tf.cast(image, tf.float32)
    return label, image


dataset = tf.data.TFRecordDataset("MNIST_data/test.tfrecords")
dataset = dataset.repeat()  # 重复此数据集
dataset = dataset.map(__format)  # 数据格式化

dataset = dataset.batch(10000)  # 批次量 每次输出的个数

iterator = dataset.make_one_shot_iterator()
# label, image = iterator.get_next()  # 获取

with tf.Session() as sess:
    test_label, test_image = sess.run(iterator.get_next())
    # print(test_label)

with tf.Session(graph=tf.Graph()) as sess:
    tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.TRAINING], "model_data/")

    img = sess.graph.get_tensor_by_name("conv1/img:0")

    prob = sess.graph.get_tensor_by_name("prob:0")

    logit = sess.graph.get_tensor_by_name("logit/logit:0")
    softmax = tf.nn.softmax(logit)

    # 定义测试的准确率
    correct_prediction = tf.equal(tf.argmax(softmax, 1), tf.argmax(test_label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    acc = sess.run(accuracy, feed_dict={
        prob: 1.0,
        img: test_image
    })

    print("准确率：{0:.2f}%".format(acc*100))

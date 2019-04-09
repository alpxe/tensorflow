import tensorflow as tf
import cv2
import numpy as np


def _analyze(buffer):
    bytes = tf.decode_raw(buffer, tf.uint8)  # 字符字节转int

    label = tf.cast(tf.strided_slice(bytes, [0], [1]), dtype=tf.int32)
    label = tf.reshape(label, [])

    label = tf.one_hot(label, 10)

    image = tf.cast(tf.strided_slice(bytes, [1], [32 * 32 * 3 + 1]), tf.int32)
    image = tf.reshape(image, [3, 32, 32])
    # Convert from [depth, height, width] to [height, width, depth].
    image = tf.transpose(image, [1, 2, 0])
    return image
    pass


# 文件路径
filenames = ["cifar10_data/cifar-10-batches-bin/data_batch_{0}.bin".format(i) for i in range(1, 6)]
dataset = tf.data.FixedLengthRecordDataset(filenames, 1 + 32 * 32 * 3)

dataset = dataset.repeat()  # 此数据集会循环

# 解析一组 1+32*32*3 的数据
dataset = dataset.map(_analyze)

iterator = dataset.make_one_shot_iterator()
# label, image = iterator.get_next()

with tf.Session() as sess:
    img = sess.run(iterator.get_next())
    img = np.array(img, np.int8)
    b, g, r = cv2.split(img)
    print(b)

    a = b[0][0]
    print(a)
    print(type(a))

    cv2.imshow('w',img)
    cv2.waitKey()


    img2 = cv2.imread("ww.png")
    b2, g2, r2 = cv2.split(img2)
    print(b2)
    a2 = b2[0][0]
    print(a2)
    print(type(a2))

    pass

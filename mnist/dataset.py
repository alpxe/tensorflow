"""
将mnist构建成TFRecord
构建 train image和label
"""
import gzip
import struct
import tensorflow as tf
import numpy as np
import os

filepath = "MNIST_data/"
outpath = "tfrecords/"

TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
TEST_LABELS = 't10k-labels-idx1-ubyte.gz'


def create(IMAGES, LABELS, NAME):
    with open(os.path.join(filepath, IMAGES), 'rb') as f:  # 打开文件
        with gzip.GzipFile(fileobj=f) as bytestream:  # 解压缩
            img_buf = bytestream.read()  # 二进制内容

    with open(os.path.join(filepath, LABELS), 'rb') as f:
        with gzip.GzipFile(fileobj=f) as bytestream:
            label_buf = bytestream.read()

    magic, items, row, col = struct.unpack_from(">IIII", img_buf, 0)

    print("items: {0}".format(items))

    img_hd = struct.calcsize(">IIII")
    label_hd = struct.calcsize(">II")

    if not os.path.exists(outpath):
        os.makedirs(outpath)  # 生成文件夹

    # 生成.tfrecords
    writer = tf.python_io.TFRecordWriter(os.path.join(outpath, NAME))  # 生成的文件名
    for i in range(items):
        label, = struct.unpack_from(">B", label_buf, label_hd + i * 1)

        img = struct.unpack_from(">784B", img_buf, img_hd + i * row * col)
        img = np.array(img)

        example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
            "image": tf.train.Feature(int64_list=tf.train.Int64List(value=img))
        }))
        writer.write(example.SerializeToString())

        if i % 6000 == 0:
            print("Waiting step {0}".format(i / 6000))

    writer.close()
    print('Successfully writer.')
    pass


# 创建训练集
create(TRAIN_IMAGES, TRAIN_LABELS, "train.tfrecords")

# 创建测试集
create(TEST_IMAGES, TEST_LABELS, "test.tfrecords")

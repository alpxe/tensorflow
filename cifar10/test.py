import tensorflow as tf


# writer = tf.python_io.TFRecordWriter("cifar10_data/records/test.records")
# # example=tf.data.experimental.TFRecordWriter("cifar10_data/records/test.records")
# for i in range(50):
#     features = tf.train.Features(feature={
#         'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[i])),
#         'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes("Inception{0}".format(i), encoding="utf-8")]))
#     })
#     example = tf.train.Example(features=features)
#
#     writer.write(example.SerializeToString())
#     pass
#
# writer.close()


def _anayze(buffer):
    fats = tf.parse_single_example(buffer, features={
        "label": tf.FixedLenFeature([1], dtype=tf.int64),
        "image": tf.FixedLenFeature([1], dtype=tf.string)
    })
    return fats["label"], fats["image"]


dataset = tf.data.TFRecordDataset("cifar10_data/records/test.records")

dataset = dataset.map(_anayze)

iter = dataset.make_one_shot_iterator()

with tf.Session() as sess:
    a = sess.run(iter.get_next())
    print(a)
    a = sess.run(iter.get_next())
    print(a)
    a = sess.run(iter.get_next())
    print(a)
    pass

import tensorflow as tf


def _resolve(buf):
    fats = tf.parse_single_example(buf, features={
        "label": tf.FixedLenFeature([1], dtype=tf.int64),
        "image": tf.FixedLenFeature([28 * 28], dtype=tf.int64)
    })

    label = tf.reshape(fats["label"], shape=[])
    image = tf.reshape(fats["image"] / 0xFF, shape=[28, 28, 1])

    return label, image


data = tf.data.TFRecordDataset("tfrecords/train.tfrecords")
data = data.repeat()
data = data.map(_resolve)
data = data.batch(2)

iterator = data.make_one_shot_iterator()
label, image = iterator.get_next()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    
    pass

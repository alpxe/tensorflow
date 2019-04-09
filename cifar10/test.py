import math
import tensorflow as tf


a=tf.clip_by_value(0., 1e-10, 1.0)


with tf.Session() as sess:
    print(sess.run(a))
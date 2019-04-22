import tensorflow as tf

with tf.Session() as sess:
    a = tf.constant([[1,2], [2,3], [3,4], [4,5], [5,6]])

    sum=tf.reduce_sum(a,1)
    red=tf.reduce_mean(sum)
    print(sess.run([sum,red]))
    pass

# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 09:46:34 2018

@author: most_pan
"""

import collections
import time
from datetime import datetime
import math

import tensorflow as tf

slim = tf.contrib.slim


class Block(collections.namedtuple('Block', ['scope', 'unit_fn', 'args'])):
    'A named tuple describing a ResNet block'


# unit_fn是残差学习元生成函数
# args是一个长度等于Block中单元数目的序列，序列中的每个元素
# 包含第三层通道数，前两层通道数以及中间层步长(depth, depth_bottleneck, stride)三个量
# 在定义一个Block类的对象时，需要提供三个信息，分别是scope，残差学习单元生成函数，以及参数列表


# 降采样函数
def subsample(inputs, factor, scope=None):
    if factor == 1:
        return inputs
    else:
        return slim.max_pool2d(inputs, [1, 1], stride=factor, scope=scope)


##用于保证维度一致
def conv2d_same(inputs, num_outputs, kernel_size, stride, scope=None):
    if stride == 1:
        return slim.conv2d(inputs, num_outputs, kernel_size, stride=1, padding='SAME', scope=scope)
    else:
        pad_total = kernel_size - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        # 使用tf.pad对图像进行填充
        inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
        return slim.conv2d(inputs, num_outputs, kernel_size, stride=stride,
                           padding='VALID', scope=scope)


@slim.add_arg_scope
def stack_blocks_dense(net, blocks, outputs_collections=None):
    for block in blocks:
        with tf.variable_scope(block.scope, 'block', [net])as sc:
            for i, unit in enumerate(block.args):
                with tf.variable_scope('unit_%d' % (i + 1), values=[net]):
                    unit_depth, unit_depth_bottleneck, unit_stride = unit  # 获取每个Block中的参数，包括第三层通道数，前两层通道数以及中间层步长
                    # unit_fn是Block类的残差神经元生成函数，它按顺序创建残差学习元并进行连接
                    net = block.unit_fn(net,
                                        depth=unit_depth,
                                        depth_bottleneck=unit_depth_bottleneck,
                                        stride=unit_stride)
            net = slim.utils.collect_named_outputs(outputs_collections, sc.name, net)
    return net


def resnet_arg_scope(is_training=True,
                     weight_decay=0.0001,
                     batch_norm_decay=0.997,
                     batch_norm_epsilon=1e-5,
                     batch_norm_scale=True):
    batch_norm_params = {
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
        'fused': None,  # Use fused batch norm if possible.
    }

    with slim.arg_scope(
            [slim.conv2d],
            weights_regularizer=slim.l2_regularizer(weight_decay),
            weights_initializer=slim.variance_scaling_initializer(),
            activation_fn=tf.nn.relu,
            normalizer_fn=slim.batch_norm,
            normalizer_params=batch_norm_params):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
            with slim.arg_scope([slim.max_pool2d], padding='SAME') as arg_sc:
                return arg_sc


# 利用add_arg_scope使bottleneck函数能够直接使用slim.arg_scope设置默认参数
@slim.add_arg_scope
def bottleneck(inputs, depth, depth_bottleneck, stride,
               outputs_collections=None, scope=None):
    with tf.variable_scope(scope, 'bottleneck_v2', [inputs])as sc:
        # 获取输入的通道数目
        depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
        # 先对输入进行batch_norm，再进行非线性激活
        preact = slim.batch_norm(inputs, activation_fn=tf.nn.relu, scope='preact')

        # 如果残差神经元的输出通道数目和输入的通道数目相同，那么直接对图像进行降采样，以保证shortcut尺寸和经历三个卷积层后的输出的此存相同
        if depth == depth_in:
            shortcut = subsample(inputs, stride, 'shortcut')
        # 如果残差神经元的输出通道数目和输入的通道数目不同，利用尺寸为1x1的卷积核对输入进行卷积，使输入通道数相同；
        else:
            shortcut = slim.conv2d(preact, depth, [1, 1], stride=stride, normalizer_fn=None,
                                   activation_fn=None, scope='shortcut')
        # 然后，定义三个卷积层
        residual = slim.conv2d(preact, depth_bottleneck, [1, 1], stride=1, scope='conv1')
        residual = conv2d_same(residual, depth_bottleneck, 3, stride, scope='conv2')
        residual = slim.conv2d(residual, depth, [1, 1], stride=1, normalizer_fn=None, activation_fn=None, scope='conv3')

        # 将shortcut和residual相加，作为输出
        output = shortcut + residual

        return slim.utils.collect_named_outputs(outputs_collections, sc.name, output)


# input是输入，blocks包含残差学习元的参数，num_classes是输出分类数，global_pool是是否进行平均池化的标志位；
def resnet_v2(inputs,
              blocks,
              num_classes=None,
              global_pool=True,
              include_root_block=True,
              reuse=None,
              scope=None):
    with tf.variable_scope(scope, 'resnet_v2', [inputs], reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, bottleneck, stack_blocks_dense],
                            outputs_collections=end_points_collection):
            net = inputs
            if include_root_block:
                with slim.arg_scope([slim.conv2d], activation_fn=None, normalizer_fn=None):
                    # 卷积核为7x7步长为2的卷积层
                    net = conv2d_same(net, 64, 7, stride=2, scope='conv1')
                # 最大值池化
                net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1')
            # 调用stack_blocks_dense堆叠50个残差学习元，每个有三个卷积层
            net = stack_blocks_dense(net, blocks)
            # 先做batch norm然后使用relu激活
            net = slim.batch_norm(net, activation_fn=tf.nn.relu, scope='postnorm')
            if global_pool:  # 进行平均池化
                net = tf.reduce_mean(net, [1, 2], name='pool5', keep_dims=True)
            # 一个输出为num_classes的卷积层，不进行激活也不归一正则化。
            if num_classes is not None:
                net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='logits')
            end_points = slim.utils.convert_collection_to_dict(end_points_collection)

            # 使用softmax进行分类
            if num_classes is not None:
                end_points['predictions'] = slim.softmax(net, scope='predictions')
            return net, end_points


# 152层残差网络
def resnet_v2_152(inputs,
                  num_classes=None,
                  global_pool=True,
                  reuse=None,
                  scope='resnet_v2_152'):
    blocks = [Block('block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
              Block('block2', bottleneck, [(512, 128, 1)] * 7 + [(512, 128, 2)]),
              Block('block3', bottleneck, [(1024, 256, 1)] * 35 + [(1024, 256, 2)]),
              Block('block4', bottleneck, [(2048, 512, 1)] * 3)]

    return resnet_v2(inputs, blocks, num_classes, global_pool,
                     include_root_block=True, reuse=reuse, scope=scope)


# 测试性能定义的函数
def time_tensorflow_run(session, target, info_string):
    num_steps_burn_in = 10
    total_duration = 0.0
    total_duration_squared = 0.0

    for i in range(num_batches + num_steps_burn_in):
        start_time = time.time()
        _ = session.run(target)
        duration = time.time() - start_time
        if i >= num_steps_burn_in:
            if not i % 10:
                print('%s: step %d, duration = %.3f' % (datetime.now(), i - num_steps_burn_in, duration))
            total_duration += duration
            total_duration_squared += duration * duration

    mn = total_duration / num_batches
    vr = total_duration_squared / num_batches - mn * mn
    sd = math.sqrt(vr)
    print('%s: %s across %d steps, %.3f +/- %.3f sec / batch' % (datetime.now(), info_string, num_batches, mn, sd))


batch_size, height, width = 32, 224, 224

inputs = tf.random_uniform((batch_size, height, width, 3))
with slim.arg_scope(resnet_arg_scope(is_training=False)):
    net, end_points = resnet_v2_152(inputs, 1000)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
num_batches = 100
time_tensorflow_run(sess, net, "ForWard")

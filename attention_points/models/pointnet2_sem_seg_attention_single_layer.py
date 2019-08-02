"""
PointNet++ Model using attention only for the first layer and max-pooling for all others
"""

import tensorflow as tf

from attention_points.attention_scannet.attention_layer import pointnet_sa_module_attention
from pointnet2_tensorflow.utils import tf_util
from pointnet2_tensorflow.utils.pointnet_util import pointnet_sa_module, pointnet_fp_module
from typing import List


def pointnet_sa_wrapper(args: List, attention=False) -> [tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Wraps the pointnet_sa module and depending on the attention flag either uses the max-pooling or the attention
    layer

    :param args: Arguments to be supplied to the pointnet_sa module
    :param attention: Whether or not attention should be used
    :return: xyz and points after applying the pointNet++ sa module as well as their indices
    """
    if attention:
        xyz, points, indices = pointnet_sa_module_attention(*args)
    else:
        xyz, points, indices = pointnet_sa_module(*args)
    return xyz, points, indices


def get_model(point_cloud: tf.Tensor, attention_layer_idx: int, is_training: tf.Variable, num_class: int,
              bn_decay=None) -> [tf.Tensor,
                                 tf.Tensor]:
    """
    Return a PointNet++ model using Attention instead of the max-pooling operations for a single layer.
    This layer is specified with the attention_layer_idx (0-3)

    :param point_cloud: Input points for the model (BxNx3)
    :param attention_layer_idx: Id of the layer that should be replaced by an attention layer (0-3)
    :param is_training: Flag whether or not the parameters should be trained or not
    :param num_class: Number of classes (e.g. 21 for ScanNet)
    :param bn_decay: BatchNorm decay
    :return: predictions for each point (B x N x num_class)
    """
    assert (0 <= attention_layer_idx < 4)
    end_points = {}
    l0_xyz = point_cloud
    l0_points = None
    end_points['l0_xyz'] = l0_xyz

    # Instead of the max-pooling we use here attention, but only for the specified layer
    params_l1 = [l0_xyz, l0_points, 1024, 0.1, 32, [32, 32, 64], None, False, is_training, bn_decay, 'layer1']
    l1_xyz, l1_points, l1_indices = pointnet_sa_wrapper(params_l1, attention_layer_idx == 0)
    params_l2 = [l1_xyz, l1_points, 256, 0.2, 32, [64, 64, 128], None, False, is_training, bn_decay, 'layer2']
    l2_xyz, l2_points, l2_indices = pointnet_sa_wrapper(params_l2, attention_layer_idx == 1)
    params_l3 = [l2_xyz, l2_points, 64, 0.4, 32, [128, 128, 256], None, False, is_training, bn_decay, 'layer3']
    l3_xyz, l3_points, l3_indices = pointnet_sa_wrapper(params_l3, attention_layer_idx == 2)
    params_l4 = [l3_xyz, l3_points, 16, 0.8, 32, [256, 256, 512], None, False, is_training, bn_decay, 'layer4']
    l4_xyz, l4_points, l4_indices = pointnet_sa_wrapper(params_l4, attention_layer_idx == 3)

    # Feature Propagation layers
    l3_points = pointnet_fp_module(l3_xyz, l4_xyz, l3_points, l4_points, [256, 256], is_training, bn_decay,
                                   scope='fa_layer1')
    l2_points = pointnet_fp_module(l2_xyz, l3_xyz, l2_points, l3_points, [256, 256], is_training, bn_decay,
                                   scope='fa_layer2')
    l1_points = pointnet_fp_module(l1_xyz, l2_xyz, l1_points, l2_points, [256, 128], is_training, bn_decay,
                                   scope='fa_layer3')
    l0_points = pointnet_fp_module(l0_xyz, l1_xyz, l0_points, l1_points, [128, 128, 128], is_training, bn_decay,
                                   scope='fa_layer4')

    # Full connected layers
    net = tf_util.conv1d(l0_points, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='fc1',
                         bn_decay=bn_decay)
    end_points['feats'] = net
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp1')
    net = tf_util.conv1d(net, num_class, 1, padding='VALID', activation_fn=None, scope='fc2')

    return net, end_points


if __name__ == '__main__':
    with tf.Graph().as_default():
        points = tf.zeros((16, 8192, 3))
        pred, _ = get_model(points, 0, tf.constant(True), 21)
        print(pred)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        res = sess.run(pred)
        print(res, res.shape)

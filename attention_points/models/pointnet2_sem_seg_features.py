"""
PointNet++ Model using the additional features
"""

import tensorflow as tf

from pointnet2_tensorflow.utils import tf_util
from pointnet2_tensorflow.utils.pointnet_util import pointnet_sa_module, pointnet_fp_module


def get_model(point_cloud: tf.Tensor, features: tf.Tensor, is_training: tf.Variable, num_class: int, bn_decay=None) -> \
        [tf.Tensor, tf.Tensor]:
    """
    Return a PointNet++ model using additional features as input for the first layer

    :param point_cloud: Input points for the model (BxNx3)
    :param features: The features for each point (BxNxk)
    :param is_training: Flag whether or not the parameters should be trained or not
    :param num_class: Number of classes (e.g. 21 for ScanNet)
    :param bn_decay: BatchNorm decay
    :return: predictions for each point (B x N x num_class)
    """
    end_points = {}
    l0_xyz = point_cloud
    l0_points = features
    end_points['l0_xyz'] = l0_xyz

    # Layer 1
    l1_xyz, l1_points, l1_indices = pointnet_sa_module(l0_xyz, l0_points, npoint=1024, radius=0.1, nsample=32,
                                                       mlp=[32, 32, 64], mlp2=None, group_all=False,
                                                       is_training=is_training, bn_decay=bn_decay, scope='layer1')
    l2_xyz, l2_points, l2_indices = pointnet_sa_module(l1_xyz, l1_points, npoint=256, radius=0.2, nsample=32,
                                                       mlp=[64, 64, 128], mlp2=None, group_all=False,
                                                       is_training=is_training, bn_decay=bn_decay, scope='layer2')
    l3_xyz, l3_points, l3_indices = pointnet_sa_module(l2_xyz, l2_points, npoint=64, radius=0.4, nsample=32,
                                                       mlp=[128, 128, 256], mlp2=None, group_all=False,
                                                       is_training=is_training, bn_decay=bn_decay, scope='layer3')
    l4_xyz, l4_points, l4_indices = pointnet_sa_module(l3_xyz, l3_points, npoint=16, radius=0.8, nsample=32,
                                                       mlp=[256, 256, 512], mlp2=None, group_all=False,
                                                       is_training=is_training, bn_decay=bn_decay, scope='layer4')

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
        inputs = tf.zeros((32, 2048, 3))
        net, _ = get_model(inputs, tf.constant(True), 10)
        print(net)

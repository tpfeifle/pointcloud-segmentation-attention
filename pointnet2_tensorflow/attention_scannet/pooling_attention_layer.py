import tensorflow as tf
from typing import List
from utils.pointnet_util import sample_and_group
from utils import tf_util
from attention_scannet.attention_layer import AttentionLayer

class PoolingAttentionNetLayer(tf.keras.layers.Layer):
    def __init__(self, bn_decay, mlp, npoint: int, out_dim: int, is_training: bool = True,
                 radius: float = 0.1, nsample: int = 32, bn: bool = True):
        """
        :param npoint: number of groups to sample
        :param radius:  radius of the ball query
        :param nsample: number of samples in a group
        :param out_dim: output dimension for each point vector
        """
        super().__init__()
        key_dim = out_dim
        self.attention_layer = AttentionLayer(out_dim, key_dim)
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.is_training = is_training
        self.bn = bn
        self.bn_decay = bn_decay
        self.mlp = mlp

    def call(self, inputs, **kwargs):

        xyz, points = inputs
        # Sample and Grouping
        new_xyz, new_points, idx, grouped_xyz = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points, False, True)

        # Point Feature Embedding
        for i, num_out_channel in enumerate(self.mlp):
            new_points = tf_util.conv2d(new_points, num_out_channel, [1, 1],
                                        padding='VALID', stride=[1, 1],
                                        bn=self.bn, is_training=self.is_training,
                                        scope='conv%d' % (i), bn_decay=self.bn_decay,
                                        data_format='NHWC')

        # Pooling using Attention
        new_points = self.attention_layer([new_points, new_xyz])


        # [Optional] Further Processing
        new_points = tf.squeeze(new_points, [2])  # (batch_size, npoints, mlp2[-1])
        return new_xyz, new_points, idx

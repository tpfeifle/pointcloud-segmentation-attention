import tensorflow as tf

from utils import tf_util
from utils.pointnet_util import pointnet_fp_module, sample_and_group, sample_and_group_all
from attention_scannet.attention_layer import AttentionLayer, AttentionNetLayer


def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size, num_point))
    smpws_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point))
    return pointclouds_pl, labels_pl, smpws_pl


class AttentionNetModel(tf.keras.layers.Layer):
    def __init__(self, is_training, bn_decay, num_class):
        super().__init__()
        self.l1 = AttentionNetLayer(npoint=1024, radius=0.1, nsample=32, out_dim=64, is_training=is_training,
                                    bn_decay=bn_decay)
        self.l2 = AttentionNetLayer(256, 0.1, 32, 128, is_training, bn_decay)
        self.l3 = AttentionNetLayer(64, 0.1, 32, 256, is_training, bn_decay)
        self.l4 = AttentionNetLayer(16, 0.1, 32, 512, is_training, bn_decay)
        self.is_training = is_training
        self.bn_decay = bn_decay
        self.num_class = num_class

    def get_end_points(self, inputs):
        end_points = {}
        l0_xyz = inputs
        l0_points = None
        l1_xyz, l1_points, l1_indices = self.l1([l0_xyz, None])
        l2_xyz, l2_points, l2_indices = self.l2([l1_xyz, l1_points])
        l3_xyz, l3_points, l3_indices = self.l3([l2_xyz, l2_points])
        l4_xyz, l4_points, l4_indices = self.l4([l3_xyz, l3_points])
        end_points['l0_xyz'] = l0_xyz

        # Feature Propagation layers
        l3_points = pointnet_fp_module(l3_xyz, l4_xyz, l3_points, l4_points, [256, 256], self.is_training,
                                       self.bn_decay, scope='fa_layer1')
        l2_points = pointnet_fp_module(l2_xyz, l3_xyz, l2_points, l3_points, [256, 256], self.is_training,
                                       self.bn_decay, scope='fa_layer2')
        l1_points = pointnet_fp_module(l1_xyz, l2_xyz, l1_points, l2_points, [256, 128], self.is_training,
                                       self.bn_decay, scope='fa_layer3')
        l0_points = pointnet_fp_module(l0_xyz, l1_xyz, l0_points, l1_points, [128, 128, 128], self.is_training,
                                       self.bn_decay, scope='fa_layer4')

        # FC layers
        net = tf_util.conv1d(l0_points, 128, 1, padding='VALID', bn=True, is_training=self.is_training, scope='fc1',
                             bn_decay=self.bn_decay)
        end_points['feats'] = net

        return end_points

    def call(self, inputs, **kwargs):
        l0_xyz = inputs
        l0_points = None
        l1_xyz, l1_points, l1_indices = self.l1([l0_xyz, tf.zeros([0])])
        print("l1_points shape: ", l1_points.shape)
        l2_xyz, l2_points, l2_indices = self.l2([l1_xyz, l1_points])
        print("l2_points shape: ", l2_points.shape)
        l3_xyz, l3_points, l3_indices = self.l3([l2_xyz, l2_points])
        print("l3_points shape: ", l3_points.shape)
        l4_xyz, l4_points, l4_indices = self.l4([l3_xyz, l3_points])
        print("l4_points shape: ", l4_points.shape)

        # Feature Propagation layers
        l3_points = pointnet_fp_module(l3_xyz, l4_xyz, l3_points, l4_points, [256, 256], self.is_training,
                                       self.bn_decay, scope='fa_layer1')
        l2_points = pointnet_fp_module(l2_xyz, l3_xyz, l2_points, l3_points, [256, 256], self.is_training,
                                       self.bn_decay, scope='fa_layer2')
        l1_points = pointnet_fp_module(l1_xyz, l2_xyz, l1_points, l2_points, [256, 128], self.is_training,
                                       self.bn_decay, scope='fa_layer3')
        l0_points = pointnet_fp_module(l0_xyz, l1_xyz, l0_points, l1_points, [128, 128, 128], self.is_training,
                                       self.bn_decay, scope='fa_layer4')

        # FC layers
        net = tf_util.conv1d(l0_points, 128, 1, padding='VALID', bn=True, is_training=self.is_training, scope='fc1',
                             bn_decay=self.bn_decay)
        net = tf_util.dropout(net, keep_prob=0.5, is_training=self.is_training, scope='dp1')
        out = tf_util.conv1d(net, self.num_class, 1, padding='VALID', activation_fn=None, scope='fc2')

        return out

    def placeholder_inputs(self, batch_size, num_point):
        pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3), name="pointclouds")
        labels_pl = tf.placeholder(tf.int32, shape=(batch_size, num_point), name="labels")
        smpws_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point), name="smpws")
        return pointclouds_pl, labels_pl, smpws_pl


if __name__ == '__main__':
    with tf.Graph().as_default():
        model = AttentionNetModel(tf.constant(True), None, 10)
        inputs = tf.zeros((32, 2048, 3))
        net = model(inputs)
        print(net)

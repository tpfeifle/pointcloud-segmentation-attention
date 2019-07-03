from typing import List

import tensorflow as tf

from utils.pointnet_util import sample_and_group


class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, output_dim, key_dim):
        super(AttentionLayer, self).__init__(name="ScannetAttentionLayer")
        self.output_dim = output_dim
        self.key_dim = key_dim

    def build(self, input_shape):
        self.query_net = tf.layers.Dense(self.key_dim)
        self.key_net = tf.layers.Dense(self.key_dim)
        self.value_net = tf.layers.Dense(self.output_dim)

    def call(self, inputs, **kwargs):
        input, query = inputs
        Q = self.query_net(query)
        Q = tf.expand_dims(Q, axis=2)
        K = self.key_net(input)
        V = self.value_net(input)
        weights = tf.matmul(Q, K, transpose_b=True)
        weights = weights / tf.sqrt(tf.to_float(self.key_dim))
        weights = tf.nn.softmax(weights, dim=-1)
        out = tf.matmul(weights, V)
        out = tf.squeeze(out, axis=2)
        return out


class InnerAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, output_dim, key_dim):
        super(InnerAttentionLayer, self).__init__(name="ScannetInnerAttentionLayer")
        self.output_dim = output_dim
        self.key_dim = key_dim

        self.num_heads = 4

    def build(self, input_shape):
        self.query_net = tf.layers.Dense(self.key_dim, input_shape=(3,))
        self.key_net = tf.layers.Dense(self.key_dim, input_shape=(input_shape[-1],))
        self.value_net = tf.layers.Dense(self.output_dim, input_shape=(input_shape[-1],))

        self.dense = tf.keras.layers.Dense(self.output_dim)

    def call(self, input, **kwargs):
        # here the we create a query vector for each point
        Q = self.query_net(input)
        K = self.key_net(input)
        V = self.value_net(input)

        batch_size = tf.shape(Q)[1]
        group_num = tf.shape(Q)[2]
        Q = self.split_heads(Q, batch_size, group_num, self.key_dim // self.num_heads)
        K = self.split_heads(K, batch_size, group_num, self.key_dim // self.num_heads)
        V = self.split_heads(V, batch_size, group_num, self.output_dim // self.num_heads)

        weights = tf.matmul(Q, K, transpose_b=True)
        weights = weights / tf.sqrt(tf.to_float(self.key_dim))
        weights = tf.nn.softmax(weights, dim=-1)
        out = tf.matmul(weights, V)

        scaled_attention = tf.transpose(out, perm=[0, 1, 2, 4, 3, 5])

        concat_attention = tf.reshape(scaled_attention, (1, batch_size, group_num, -1, self.output_dim))

        output = self.dense(concat_attention)

        out = tf.squeeze(output, axis=0)
        return out

    def split_heads(self, x, batch_size, group_num, depth):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (1, batch_size, group_num, -1, self.num_heads, depth))
        return tf.transpose(x, perm=[0, 1, 2, 4, 3, 5])


class FeedForwardLayer(tf.keras.layers.Layer):
    def __init__(self, input_and_output_dim, inner_dim, dropout=0):
        super(FeedForwardLayer, self).__init__(name="ScannetFeedForwardLayer")
        self.input_and_output_dim = input_and_output_dim
        self.inner_dim = inner_dim
        self.dropout = dropout

    def build(self, _):
        self.layer_1 = tf.layers.Dense(self.inner_dim)
        self.layer_2 = tf.layers.Dense(self.input_and_output_dim)

    def call(self, input, **kwargs):
        x = self.layer_1(input)
        x = tf.nn.relu(x)
        x = tf.layers.dropout(x, rate=self.dropout)
        x = self.layer_2(x)
        return x


class InnerAttentionBlock(tf.keras.layers.Layer):
    def __init__(self, out_dim, key_dim):
        super(InnerAttentionBlock, self).__init__(name="ScannetInnerAttentionBlock")
        self.out_dim = out_dim
        self.key_dim = key_dim
        self.attention_layer = InnerAttentionLayer(out_dim, key_dim)
        self.feed_forward_layer = FeedForwardLayer(out_dim, out_dim)
        self.pre_feed_forward_layer = FeedForwardLayer(out_dim, out_dim)

    def call(self, input, **kwargs):
        points = input
        points = self.pre_feed_forward_layer(points)
        points = self.attention_layer(points)
        # TODO skip connection (if possible)
        # TODO batchnorm
        points = self.feed_forward_layer(points) + points
        # TODO batchnorm
        return points


class AttentionNetLayer(tf.keras.layers.Layer):
    def __init__(self, npoint: int, out_dim: int, inner_dimensions: List[int], is_training: bool = True,
                 radius: float = 0.1, nsample: int = 32, bn: bool = True):
        """

        :param npoint: number of groups to sample
        :param radius:  radius of the ball query
        :param nsample: number of samples in a group
        :param out_dim: output dimension for each point vector
        :param is_training:
        :param bn:
        """
        super().__init__()
        key_dim = out_dim
        self.attention_layer = AttentionLayer(out_dim, key_dim)
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.is_training = is_training
        self.bn = bn
        self.inner_dimensions = inner_dimensions
        self.inner_blocks = [InnerAttentionBlock(i, key_dim) for i in inner_dimensions]

    def call(self, inputs, **kwargs):
        # Sample and Grouping
        xyz, points = inputs
        # if points == tf.zeros([0]):
        #     print("hello")
        #     points = None
        # print("before sample and group")
        new_xyz, new_points, idx, grouped_xyz = sample_and_group(self.npoint, self.radius, self.nsample, xyz,
                                                                 points, False, True)
        # print("after sample and group")
        # print(f"new_xyz: {new_xyz.shape}, new_points: {new_points.shape}\n"
        #       f"idx: {idx.shape}, grouped_xyz: {grouped_xyz.shape}\n"
        #       f"xyz: {xyz.shape}, points: {points.shape}")
        # Point Feature Embedding
        # print(f"shape of new_points: {new_points.shape}")
        second_dim = new_points.shape[1]

        for inner_block in self.inner_blocks:
            new_points = inner_block([new_points])

            # print(f"done {inner_layer}")
            # print(f"shape of new_points: {new_points.shape}")
        # print("inner layers done")
        # print(f"points: {points.shape}, new_points: {new_points.shape}")
        # TODO get not only coordinates of point as query vector, but the feature vector new_point:
        # new_points = self.attention_layer([new_points, new_points[:, :, 0, :]])
        new_points = self.attention_layer(
            [new_points, tf.zeros((tf.shape(new_points)[0], second_dim, self.inner_dimensions[-1]))])
        # print("done end layer")
        # print(f"shape of new_points: {new_points.shape}")

        # new_points = tf.squeeze(new_points, [2]) # (batch_size, npoints, mlp2[-1]) # TODO check if this line is needed
        # TODO do we want to copy the coordinates to the resulting vectors?
        #  (don't think so, sample_and_group() does this?)
        return [new_xyz, new_points, idx]


# TODO add multi head attention


if __name__ == '__main__':
    # layer = AttentionLayer(3, 13)
    # sess = tf.Session()
    # res = layer(tf.random_normal([6, 3]))
    # init_op = tf.global_variables_initializer()
    # sess.run(init_op)
    # print(sess.run(res))
    layer = AttentionNetLayer(npoint=1024, out_dim=64, inner_dimensions=[128, 96], radius=0.1, nsample=32,
                              is_training=True)
    sess = tf.Session()
    input1 = tf.zeros((32, 2048, 3))
    input2 = tf.zeros((0,))
    inputs = [input1, input2]
    res = layer(inputs)
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print(sess.run(res))

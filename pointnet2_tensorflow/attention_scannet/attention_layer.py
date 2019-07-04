from typing import List

import tensorflow as tf

from utils.pointnet_util import sample_and_group


class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, output_dim, key_dim):
        super(AttentionLayer, self).__init__(name="ScannetAttentionLayer")
        self.output_dim = output_dim
        self.key_dim = key_dim
        self.num_heads = 16

    def build(self, input_shape):
        self.query_net = tf.layers.Dense(self.key_dim * self.num_heads)
        self.key_net = tf.layers.Dense(self.key_dim * self.num_heads)
        self.value_net = tf.layers.Dense(self.output_dim * self.num_heads)
        self.out_net = tf.layers.Dense(self.output_dim)

    def call(self, inputs, **kwargs):
        input, query = inputs
        Q = self.query_net(query)
        Q = tf.expand_dims(Q, axis=2)
        K = self.key_net(input)
        V = self.value_net(input)
        Q, K, V = [tf.reshape(x, (tf.shape(x)[0], x.shape[1], self.num_heads, x.shape[2], self.key_dim))
                   for x in [Q, K, V]]
        weights = tf.matmul(Q, K, transpose_b=True)
        weights = weights / tf.sqrt(tf.to_float(self.key_dim))
        weights = tf.nn.softmax(weights, dim=-1)
        out = tf.matmul(weights, V)
        # concat heads
        concat_attention = tf.reshape(out, [tf.shape(out)[0], out.shape[1], self.num_heads * self.key_dim])
        out = self.out_net(concat_attention)
        return out


class InnerAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, output_dim, key_dim):
        super(InnerAttentionLayer, self).__init__(name="ScannetInnerAttentionLayer")
        self.output_dim = output_dim
        self.key_dim = key_dim
        self.num_heads = 4

    def build(self, input_shape):
        self.query_net = tf.layers.Dense(self.key_dim * self.num_heads)
        self.key_net = tf.layers.Dense(self.key_dim * self.num_heads)
        self.value_net = tf.layers.Dense(self.key_dim * self.num_heads)
        self.out_net = tf.keras.layers.Dense(self.output_dim)

    def call(self, input, **kwargs):
        # here the we create a query vector for each point
        Q = self.query_net(input)
        K = self.key_net(input)
        V = self.value_net(input)
        # reshape for multi-head attention
        Q, K, V = [tf.reshape(x, (1, tf.shape(x)[1], x.shape[2], x.shape[3], self.num_heads, self.key_dim))
                   for x in [Q, K, V]]

        weights = tf.matmul(Q, K, transpose_b=True)
        weights = weights / tf.sqrt(tf.to_float(self.key_dim))
        weights = tf.nn.softmax(weights, dim=-1)
        out = tf.matmul(weights, V)
        # concat heads
        concat_attention = tf.reshape(out,
                                      [tf.shape(out)[1], out.shape[2], out.shape[3], self.num_heads * self.key_dim])
        out = self.out_net(concat_attention)
        return out


class FeedForwardLayer(tf.keras.layers.Layer):
    def __init__(self, input_and_output_dim, inner_dim, dropout=0):
        super(FeedForwardLayer, self).__init__(name="ScannetFeedForwardLayer")
        self.input_and_output_dim = input_and_output_dim
        self.inner_dim = inner_dim
        self.dropout = dropout

    def build(self, _):
        self.layer_1 = tf.layers.Dense(self.inner_dim)
        self.layer_2 = tf.layers.Dense(self.inner_dim)
        self.layer_3 = tf.layers.Dense(self.inner_dim)
        self.layer_4 = tf.layers.Dense(self.input_and_output_dim)

    def call(self, input, **kwargs):
        x = self.layer_1(input)
        x = tf.nn.relu(x)
        x = tf.layers.dropout(x, rate=self.dropout)
        x = self.layer_2(x)
        x = tf.nn.relu(x)
        x = tf.layers.dropout(x, rate=self.dropout)
        x = self.layer_3(x)
        x = tf.nn.relu(x)
        x = tf.layers.dropout(x, rate=self.dropout)
        x = self.layer_4(x)
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
        new_xyz, new_points, idx, grouped_xyz = sample_and_group(self.npoint, self.radius, self.nsample, xyz,
                                                                 points, False, True)
        second_dim = new_points.shape[1]

        for inner_block in self.inner_blocks:
            new_points = inner_block([new_points])

        # TODO get not only coordinates of point as query vector, but the feature vector new_point:
        # new_points = self.attention_layer(
        #     [new_points, tf.zeros((tf.shape(new_points)[0], second_dim, self.inner_dimensions[-1]))])
        new_points = self.attention_layer(
            [new_points, tf.gather(new_points, [0], axis=2)])
        return [new_xyz, new_points, idx]


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

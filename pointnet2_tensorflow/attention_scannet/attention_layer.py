from typing import List

import tensorflow as tf

from utils.pointnet_util import sample_and_group


class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, output_dim, key_dim):
        super(AttentionLayer, self).__init__(name="ScannetAttentionLayer")
        self.output_dim = output_dim
        self.key_dim = key_dim

    def build(self, input_shape):
        self.query_net = tf.layers.Dense(self.key_dim, input_shape=(3,))
        # self.add_variable(self.query_net.variables)
        self.key_net = tf.layers.Dense(self.key_dim, input_shape=(input_shape[-1],))
        self.value_net = tf.layers.Dense(self.output_dim, input_shape=(input_shape[-1],))
        # print(input_shape[-2])
        # self.softmax = tf.nn.softmax(input_shape[-2])

    # query is all points here:
    # def call(self, input, **kwargs):
    #     Q = self.query_net(input)
    #     K = self.key_net(input)
    #     V = self.value_net(input)
    #     print("input shape ", input.shape)
    #     print("key dim: ", self.key_dim)
    #     print("Q shape: ", Q.shape)
    #     print("K shape: ", K.shape)
    #     print("V shape: ", V.shape)
    #     weights = tf.matmul(Q, K, transpose_b=True)
    #     weights = weights / tf.sqrt(tf.to_float(self.key_dim))
    #     print("weights shape: ", weights.shape)
    #     print("softmax denom shape: ", tf.expand_dims(tf.reduce_sum(tf.exp(weights), axis=-1), axis=3))
    #     # weights = tf.exp(weights) / tf.expand_dims(tf.reduce_sum(tf.exp(weights), axis=-1),axis=3) # TODO is axis correct?
    #     weights = tf.nn.softmax(weights, dim=-1)
    #     print("final weights shape: ", weights.shape)
    #     out = tf.matmul(weights, V)
    #     print("weighted sum shape: ", out.shape)
    #     return out

    def call(self, inputs, **kwargs):
        # this time the query will be only one point
        input, query = inputs
        Q = self.query_net(query)
        Q = tf.expand_dims(Q, axis=2)
        K = self.key_net(input)
        V = self.value_net(input)
        # print("input shape ", input.shape)
        # print("query shape ", query.shape)
        # print("key dim: ", self.key_dim)
        # print("Q shape: ", Q.shape)
        # print("K shape: ", K.shape)
        # print("V shape: ", V.shape)
        weights = tf.matmul(Q, K, transpose_b=True)
        weights = weights / tf.sqrt(tf.to_float(self.key_dim))
        # print("weights shape: ", weights.shape)
        weights = tf.nn.softmax(weights, dim=-1)
        # print("final weights shape: ", weights.shape)
        out = tf.matmul(weights, V)
        out = tf.squeeze(out, axis=2)
        # print("weighted sum shape: ", out.shape)
        return out


class InnerAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, output_dim, key_dim):
        super(InnerAttentionLayer, self).__init__(name="ScannetAttentionLayer")
        self.output_dim = output_dim
        self.key_dim = key_dim

    def build(self, input_shape):
        self.query_net = tf.layers.Dense(self.key_dim, input_shape=(3,))
        self.key_net = tf.layers.Dense(self.key_dim, input_shape=(input_shape[-1],))
        self.value_net = tf.layers.Dense(self.output_dim, input_shape=(input_shape[-1],))

    def call(self, input, **kwargs):
        # here the we create a query vector for each point
        Q = self.query_net(input)
        K = self.key_net(input)
        V = self.value_net(input)
        weights = tf.matmul(Q, K, transpose_b=True)
        weights = weights / tf.sqrt(tf.to_float(self.key_dim))
        weights = tf.nn.softmax(weights, dim=-1)
        out = tf.matmul(weights, V)
        out = tf.squeeze(out, axis=0)
        return out


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
        self.inner_layers = [InnerAttentionLayer(i, key_dim) for i in inner_dimensions]

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

        # Point Feature Embedding
        # print(f"shape of new_points: {new_points.shape}")
        for inner_layer in self.inner_layers:
            new_points = inner_layer([new_points])
            # print(f"done {inner_layer}")
            # print(f"shape of new_points: {new_points.shape}")
        # TODO get not only coordinates of point as query vector, but the feature vector new_point:
        new_points = self.attention_layer([new_points, new_xyz])
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

import tensorflow as tf

from attention_scannet.attention_models import AttentionNetModel
from scannet_dataset import data_transformation


def train(epochs=1, batchsize=16):
    tf.Graph().as_default()
    tf.device('/gpu:0')
    sess = tf.Session()
    # define dataset
    dataset = data_transformation.get_transformed_dataset("train")
    dataset = dataset.batch(batchsize)
    iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
    dataset_init = iterator.make_initializer(dataset)
    sess.run(dataset_init)
    next_element = iterator.get_next()
    points, labels, colors, normals, sample_weight = next_element
    # define model and metrics
    is_training_pl = tf.constant(True, tf.bool, shape=(), name="is_training")
    model = AttentionNetModel(is_training=is_training_pl, bn_decay=None, num_class=21)
    prediction = model(points)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=prediction, weights=sample_weight)

    correct = tf.equal(tf.argmax(prediction, 2), tf.to_int64(labels))
    accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(batchsize * 8192)
    optimizer = tf.train.AdamOptimizer(1e-3)
    train_op = optimizer.minimize(loss)

    variable_init = tf.global_variables_initializer()
    sess.run(variable_init)

    for i in range(100):
        _, loss_val, pred_val, accuracy = sess.run([train_op, loss, prediction, accuracy])
        print(f"loss: {loss_val}, \taccuracy: {accuracy}")


if __name__ == '__main__':
    train()

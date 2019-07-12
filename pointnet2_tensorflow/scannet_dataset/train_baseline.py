import tensorflow as tf

from attention_scannet.attention_models import AttentionNetModel
from scannet_dataset import precompute_dataset
import importlib
import os
import time

N_POINTS = 8192
N_TRAIN_SAMPLES = 1201
N_VAL_SAMPLES = 312
BATCH_SIZE = 20
MODEL = importlib.import_module("models.pointnet2_sem_seg")
LOG_DIR = os.path.join('/tmp/pycharm_project_250/pointnet2_tensorflow/log/baseline/%s' % int(time.time()))


def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
        1e-3,  # Base learning rate.
        tf.multiply(batch, BATCH_SIZE),  # Current index into the dataset. batch * BATCH_SIZE
        200000,  # Decay step.
        0.7,  # Decay rate.
        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001)  # CLIP THE LEARNING RATE!
    return learning_rate

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
        0.5,
        tf.multiply(batch, BATCH_SIZE),
        200000.0,
        0.5,
        staircase=True)
    bn_decay = tf.minimum(0.99, 1 - bn_momentum)
    return bn_decay

def train(epochs=1000, batch_size=BATCH_SIZE, n_epochs_to_val=4):
    tf.Graph().as_default()
    tf.device('/gpu:0')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    # define train
    train_data = precompute_dataset.get_precomputed_train_data_set()
    train_data = train_data.batch(batch_size).prefetch(4)
    train_iterator = tf.data.Iterator.from_structure(train_data.output_types, train_data.output_shapes)
    train_data_init = train_iterator.make_initializer(train_data)
    sess.run(train_data_init)
    points, labels, colors, train_normals, sample_weight = train_iterator.get_next()
    train_features = tf.concat([tf.cast(colors, tf.float32), train_normals], 2)
    train_coordinates = points
    train_labels = labels
    train_sample_weight = sample_weight

    # define validation data
    val_data = precompute_dataset.get_precomputed_val_data_set()
    val_data = val_data.batch(batch_size).prefetch(4)
    val_iterator = tf.data.Iterator.from_structure(val_data.output_types, train_data.output_shapes)
    val_data_init = val_iterator.make_initializer(val_data)
    sess.run(val_data_init)
    points, labels, colors, normals, sample_weight = train_iterator.get_next()
    val_features = tf.concat([tf.cast(colors, tf.float32), normals], 0)
    val_coordinates = points
    val_labels = labels
    val_sample_weight = sample_weight

    train_writer = tf.summary.FileWriter(LOG_DIR, sess.graph)




    # define model and metrics
    # is_training_pl = tf.constant(True, tf.bool, shape=(), name="is_training")
    is_training_pl = tf.Variable(True)
    step = tf.Variable(0, trainable=False)
    # model = AttentionNetModel(is_training=is_training_pl, bn_decay=None, num_class=21)

    # train_pred = model(train_coordinates)
    train_pred, _ = MODEL.get_model(train_coordinates, is_training_pl, 21, bn_decay=get_bn_decay(step))
    train_loss = tf.losses.sparse_softmax_cross_entropy(labels=train_labels, logits=train_pred, weights=train_sample_weight)

    correct_train_pred = tf.equal(tf.argmax(train_pred, 2, output_type=tf.int32), train_labels)

    train_iou, conf_mat = tf.metrics.mean_iou(train_labels, tf.argmax(train_pred, 2, output_type=tf.int32), num_classes=21)
    train_acc = tf.reduce_sum(tf.cast(correct_train_pred, tf.float32)) / float(batch_size * N_POINTS)
    bn_decay = get_learning_rate(step)
    optimizer = tf.train.AdamOptimizer(bn_decay)
    train_op = optimizer.minimize(train_loss, global_step=step)
    # val_pred = model(val_coordinates)
    '''val_loss = tf.losses.sparse_softmax_cross_entropy(labels=val_labels, logits=val_pred,
                                                      weights=val_sample_weight)
    correct_val_pred = tf.equal(tf.argmax(val_pred, 2, output_type=tf.int32), val_labels)
    val_acc = tf.reduce_sum(tf.cast(correct_val_pred, tf.float32)) / \
              tf.cast(tf.shape(labels)[0] * N_POINTS, dtype=tf.float32)'''





    variable_init = tf.global_variables_initializer()
    sess.run(variable_init)
    sess.run(tf.local_variables_initializer())

    tf.summary.scalar('accuracy', train_acc)
    tf.summary.scalar('loss', train_loss)
    tf.summary.scalar('iou', train_iou)
    tf.summary.scalar('bn_decay', bn_decay)
    batches_per_epoch = N_TRAIN_SAMPLES / batch_size
    # batches_per_epoch = 2
    print(f"batches per epoch: {batches_per_epoch}")
    assign_op = is_training_pl.assign(True)
    sess.run(assign_op)
    print(tf.trainable_variables())
    acc_sum, loss_sum = 0, 0
    for i in range(int(epochs * batches_per_epoch)):
        # step.assign(i)

        epoch = int((i + 1) / batches_per_epoch) + 1
        merged = tf.summary.merge_all()

        # _, loss_val, acc_val, pred, batch_data, iou, asdf, merged = sess.run([train_op, train_loss, train_acc, train_pred, points, train_iou, conf_mat, merged])
        # extract labels and predictions
        _, loss_val, acc_train, pred_train, labels_val, merged_val = sess.run([train_op, train_loss, train_acc, train_pred, train_labels, merged])

        if i % 50 == 0:
            # visualize for debugging
            import numpy as np
            import matplotlib.pyplot as plt
            max_pred = np.argmax(pred_train, axis=2)
            all_batches_pred = np.reshape(max_pred, -1)
            plt.hist(all_batches_pred, bins=21)
            plt.show()
            # END visualize


        acc_sum += acc_train
        loss_sum += loss_val
        print(f"\tbatch {(i + 1) % int(batches_per_epoch)}\tloss: {loss_val}, \taccuracy: {acc_train}")
        train_writer.add_summary(merged_val, i)
        if (i + 1) % int(batches_per_epoch) == 0:

            print(f"epoch {epoch} finished")
            # epoch summary
            print(f"mean acc: {acc_sum / batches_per_epoch} \tmean loss: {loss_sum / batches_per_epoch}")

            acc_sum, loss_sum = 0, 0
            '''
            if epoch % n_epochs_to_val == 0 and False:
                assign_op = is_training_pl.assign(False)
                sess.run(assign_op)
                print("starting evaluation")
                for j in range(N_VAL_SAMPLES):
                    loss_val, acc_val = sess.run([val_loss, val_acc])
                    print(f"\tscene {j} eval: \tloss: {loss_val}, \taccuracy: {acc_val}")
                    acc_sum += acc_val
                    loss_sum += loss_val
                print(f"eval: mean acc: {acc_sum / N_VAL_SAMPLES} \tmean loss: {loss_sum / N_VAL_SAMPLES}")
                acc_sum, loss_sum = 0, 0
                assign_op = is_training_pl.assign(True)
                sess.run(assign_op)'''


if __name__ == '__main__':
    train()

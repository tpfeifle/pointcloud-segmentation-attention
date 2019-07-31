import os
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import attention_points.models.pointnet2_sem_seg_attention as model
from attention_points.scannet_dataset import precompute_dataset

N_POINTS = 8192
N_TRAIN_SAMPLES = 1201 // 3
N_VAL_SAMPLES = 4542 // 3
BATCH_SIZE = 16
LOG_DIR = os.path.join(f'/home/tim/training_log/subset/temp_{int(time.time())}')

class_weights = tf.constant([0, 2.743064592944318, 3.0830506790927132, 4.785754459526457, 4.9963745147506184,
                             4.372710774561782, 5.039124880965811, 4.86451825464344, 4.717751595568025,
                             4.809412839311939, 5.052097251455304, 5.389129668645318, 5.390614085649042,
                             5.127458225110977, 5.086056870814752, 5.3831185190895265, 5.422684124268539,
                             5.422955391988761, 5.433705358072363, 5.417426773812747, 4.870172044153657])


def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
        1e-3,  # Base learning rate.
        tf.multiply(batch, BATCH_SIZE),  # Current index into the dataset. batch * BATCH_SIZE
        N_TRAIN_SAMPLES * 80,  # decay step original was 2000000, now it's after 45 epochs
        0.7,  # decay rate
        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001)  # CLIP THE LEARNING RATE!
    return learning_rate


def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
        0.5,
        tf.multiply(batch, BATCH_SIZE),
        N_TRAIN_SAMPLES * 80,  # decay step original was 2000000, now it's after 45 epochs
        0.5,
        staircase=True)
    bn_decay = tf.minimum(0.99, 1 - bn_momentum)
    return bn_decay


def show_prediction_historgram(prediction):
    # visualize for debugging
    max_pred = np.argmax(prediction, axis=2)
    all_batches_pred = np.reshape(max_pred, -1)
    plt.hist(all_batches_pred, bins=21)
    plt.show()


def train(epochs=250, batch_size=BATCH_SIZE, n_epochs_to_val=4):
    tf.Graph().as_default()
    tf.device('/gpu:0')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    # define train data
    train_data = precompute_dataset.get_precomputed_train_subset_data_set()
    train_data = train_data.batch(batch_size).prefetch(4)
    train_iterator = tf.data.Iterator.from_structure(train_data.output_types, train_data.output_shapes)
    train_data_init = train_iterator.make_initializer(train_data)
    sess.run(train_data_init)
    points, labels, colors, train_normals, sample_weight = train_iterator.get_next()
    train_features = tf.concat([tf.cast(colors, tf.float32), train_normals], 2)
    train_coordinates = points
    train_labels = labels
    # train_sample_weight = sample_weight
    train_mask = tf.not_equal(sample_weight, 0.0)
    train_mask = tf.cast(train_mask, tf.float32)
    train_sample_weight = tf.multiply(tf.gather(class_weights, train_labels), train_mask)

    # define validation data
    val_data = precompute_dataset.get_precomputed_val_subset_data_set()
    val_data = val_data.batch(batch_size).prefetch(4)
    val_iterator = tf.data.Iterator.from_structure(val_data.output_types, val_data.output_shapes)
    val_data_init = val_iterator.make_initializer(val_data)
    sess.run(val_data_init)
    points, labels, colors, normals, sample_weight = val_iterator.get_next()
    val_features = tf.concat([tf.cast(colors, tf.float32), normals], 0)
    val_coordinates = points
    val_labels = labels
    # val_sample_weight = sample_weight
    val_sample_weight = tf.gather(class_weights, val_labels)

    # define model and metrics
    is_training_pl = tf.Variable(True)
    step = tf.Variable(0, trainable=False)
    bn_decay = get_bn_decay(step)
    learning_rate = get_learning_rate(step)

    # train metrics
    train_pred, _ = model.get_model(train_coordinates, is_training_pl, 21, bn_decay=bn_decay)
    train_loss = tf.losses.sparse_softmax_cross_entropy(labels=train_labels, logits=train_pred,
                                                        weights=train_sample_weight)

    # Filter out the unassigned labels
    train_labels_flat = tf.reshape(train_labels, [-1])
    train_pred_flat = tf.reshape(train_pred, [-1, train_pred.shape[2]])

    loc = tf.reshape(tf.where(train_labels_flat > 0), [-1])
    train_labels_assigned = tf.gather(train_labels_flat, loc, axis=0)
    train_pred_assigned = tf.gather(train_pred_flat, loc)
    correct_train_pred = tf.equal(tf.argmax(train_pred_assigned, 1, output_type=tf.int32), train_labels_assigned)

    train_iou, train_iou_update = tf.metrics.mean_iou(train_labels_assigned,
                                                      tf.argmax(train_pred_assigned, 1, output_type=tf.int32),
                                                      num_classes=21, name="train_iou")
    running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="train_iou")
    train_iou_reset = tf.variables_initializer(var_list=running_vars)
    train_acc = tf.reduce_sum(tf.cast(correct_train_pred, tf.float32)) / \
                tf.cast(tf.shape(train_labels_assigned)[0], tf.float32)

    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(train_loss, global_step=step)

    # validation metrics
    val_pred, _ = model.get_model(val_coordinates, is_training_pl, 21)
    val_loss = tf.losses.sparse_softmax_cross_entropy(labels=val_labels, logits=val_pred,
                                                      weights=val_sample_weight)

    # Filter out the unassigned labels
    val_labels_flat = tf.reshape(val_labels, [-1])
    val_pred_flat = tf.reshape(val_pred, [-1, val_pred.shape[2]])

    loc = tf.reshape(tf.where(val_labels_flat > 0), [-1])
    val_labels_assigned = tf.gather(val_labels_flat, loc)
    val_pred_assigned = tf.gather(val_pred_flat, loc)
    correct_val_pred = tf.equal(tf.argmax(val_pred_assigned, 1, output_type=tf.int32), val_labels_assigned)

    val_acc = tf.reduce_sum(tf.cast(correct_val_pred, tf.float32)) / \
              tf.cast(tf.shape(val_labels_assigned)[0], tf.float32)
    val_iou, val_iou_update = tf.metrics.mean_iou(val_labels_assigned,
                                                  tf.argmax(val_pred_assigned, 1, output_type=tf.int32),
                                                  num_classes=21, name="val_iou")
    running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="val_iou")
    val_iou_reset = tf.variables_initializer(var_list=running_vars)

    # initialize variables
    variable_init = tf.global_variables_initializer()
    sess.run(variable_init)
    sess.run(tf.local_variables_initializer())
    assign_op = is_training_pl.assign(True)
    sess.run(assign_op)

    # add summaries
    train_writer = tf.summary.FileWriter(LOG_DIR + "_train", sess.graph)
    # tf.summary.scalar('accuracy', train_acc)
    # tf.summary.scalar('loss', train_loss)
    # tf.summary.scalar('iou', train_iou)
    # tf.summary.scalar('bn_decay', bn_decay)
    # tf.summary.scalar('learning_rate', learning_rate)
    # tf.summary.scalar('step', step)
    # train_summaries = tf.summary.merge_all()

    val_writer = tf.summary.FileWriter(LOG_DIR + "_val")
    # val_acc_summary = tf.summary.scalar('accuracy', val_acc)
    # val_iou_summary = tf.summary.scalar('iou', val_iou)
    # val_summaries = tf.summary.merge([val_acc_summary, val_iou_summary])

    saver = tf.train.Saver()

    batches_per_epoch = N_TRAIN_SAMPLES / batch_size
    print(f"batches per epoch: {batches_per_epoch}")
    # print(tf.trainable_variables())
    acc_sum, loss_sum = 0, 0
    best_iou = 0

    # train loop
    for i in range(int(epochs * batches_per_epoch)):
        step.assign(i)

        epoch = int((i + 1) / batches_per_epoch) + 1

        _, loss_val, acc_train, pred_train, labels_val, _, train_iou_val = sess.run(
            [train_op, train_loss, train_acc, train_pred, train_labels, train_iou_update, train_iou])

        acc_sum += acc_train
        loss_sum += loss_val
        print(f"\tepoch: {epoch:03d}\tbatch {i % int(batches_per_epoch) + 1:03d}"
              f"\tloss: {loss_val:.4f}, \taccuracy: {acc_train:.4f}\taccumulated iou: {train_iou_val:.4f}")

        if i % 50 == 0 and False:
            show_prediction_historgram(pred_train)

        if (i + 1) % int(batches_per_epoch) == 0:
            # end of epoch
            print(f"epoch {epoch} finished")
            # epoch summary
            lr, bn_d = sess.run([learning_rate, bn_decay])
            epoch_loss = loss_sum / batches_per_epoch
            epoch_acc = acc_sum / batches_per_epoch
            epoch_iou = train_iou_val
            print(f"mean loss: {epoch_loss:.4f}\tmean acc: {epoch_acc:.4f}\tmean iou: {epoch_iou:.4f}")
            summary = tf.Summary()
            summary.value.add(tag="loss", simple_value=epoch_loss)
            summary.value.add(tag="accuracy", simple_value=epoch_acc)
            summary.value.add(tag="iou", simple_value=epoch_iou)
            summary.value.add(tag="learning_rate", simple_value=lr)
            summary.value.add(tag="bn_decay", simple_value=bn_d)
            train_writer.add_summary(summary, epoch)
            # reset accumulators
            acc_sum, loss_sum = 0, 0
            sess.run(train_iou_reset)

            if epoch % n_epochs_to_val == 0:
                # pass over validation set
                assign_op = is_training_pl.assign(False)
                sess.run(assign_op)
                val_batches = N_VAL_SAMPLES // BATCH_SIZE
                print(f"starting evaluation {val_batches} batches")
                for j in range(val_batches):
                    loss_val, acc_val, _, val_iou_val = sess.run([val_loss, val_acc, val_iou_update, val_iou])
                    print(f"\tevaluation epoch: {epoch:03d}\tbatch {j:03d} eval:"
                          f"\tloss: {loss_val:.4f}\taccuracy: {acc_val:.4f}\taccumulated iou {val_iou_val:.4f}")
                    acc_sum += acc_val
                    loss_sum += loss_val
                # validation summary
                epoch_loss = loss_sum / val_batches
                epoch_acc = acc_sum / val_batches
                epoch_iou = val_iou_val
                summary = tf.Summary()
                summary.value.add(tag="loss", simple_value=epoch_loss)
                summary.value.add(tag="accuracy", simple_value=epoch_acc)
                summary.value.add(tag="iou", simple_value=epoch_iou)
                val_writer.add_summary(summary, epoch)
                print(f"evaluation:\tmean loss: {epoch_loss:.4f}\tmean acc: {epoch_acc:.4f}"
                      f"\tmean iou {epoch_iou:.4f}\n")

                # save model if it is better
                if epoch_iou > best_iou:
                    best_iou = epoch_iou
                    save_path = saver.save(sess, os.path.join(LOG_DIR + "_train", f"best_model_epoch_{epoch:03d}.ckpt"))
                    print(f"Model saved in file: {save_path}\n")

                # reset accumulators
                acc_sum, loss_sum = 0, 0
                sess.run(val_iou_reset)

                assign_op = is_training_pl.assign(True)
                sess.run(assign_op)
            print(f"starting epoch {epoch + 1}")


if __name__ == '__main__':
    train()

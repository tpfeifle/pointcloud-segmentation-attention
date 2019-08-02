import os
import time
from typing import Tuple, Optional, Callable

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from attention_points.models import pointnet2_sem_seg_features
from attention_points.scannet_dataset import precompute_dataset
from pointnet2_tensorflow.models import pointnet2_sem_seg

N_POINTS = 8192
N_TRAIN_SAMPLES = 1201  # number of train scenes
N_VAL_SAMPLES = 4542  # number of chunks in the validation set
BATCH_SIZE = 16
LOG_DIR = os.path.join('/home/tim/training_log/tmp%s' % int(time.time()))

CLASS_WEIGHTS = tf.constant([0, 2.743064592944318, 3.0830506790927132, 4.785754459526457, 4.9963745147506184,
                             4.372710774561782, 5.039124880965811, 4.86451825464344, 4.717751595568025,
                             4.809412839311939, 5.052097251455304, 5.389129668645318, 5.390614085649042,
                             5.127458225110977, 5.086056870814752, 5.3831185190895265, 5.422684124268539,
                             5.422955391988761, 5.433705358072363, 5.417426773812747, 4.870172044153657])


def get_learning_rate(batch: tf.Variable) -> tf.Variable:
    """
    computes learning rate from batch index

    :param batch: index of the current batch
    :return: learning rate
    """
    learning_rate = tf.train.exponential_decay(
        1e-3,  # Base learning rate.
        tf.multiply(batch, BATCH_SIZE),  # Current index into the dataset. batch * BATCH_SIZE
        N_TRAIN_SAMPLES * 80,  # decay step original was 2000000, now it's after 45 epochs
        0.7,  # decay rate
        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001)  # CLIP THE LEARNING RATE!
    return learning_rate


def get_bn_decay(batch: tf.Variable) -> tf.Variable:
    """
    computes batch norm decay from batch index

    :param batch: index of the current batch
    :return: batch norm decay
    """
    bn_momentum = tf.train.exponential_decay(
        0.5,
        tf.multiply(batch, BATCH_SIZE),
        N_TRAIN_SAMPLES * 80,  # decay step original was 2000000, now it's after 45 epochs
        0.5,
        staircase=True)
    bn_decay = tf.minimum(0.99, 1 - bn_momentum)
    return bn_decay


def show_prediction_historgram(prediction: np.ndarray):
    """
    visualize the predictions of the model with a histogram

    :param prediction: prediction logits
    :return:
    """
    max_pred = np.argmax(prediction, axis=2)
    all_batches_pred = np.reshape(max_pred, -1)
    plt.hist(all_batches_pred, bins=21)
    plt.show()


def get_data_tensors(data_set: tf.data.Dataset,
                     sess: tf.Session,
                     batch_size: int,
                     color: bool,
                     normal: bool) \
        -> Tuple[tf.Tensor, tf.Tensor, Optional[tf.Tensor], tf.Tensor]:
    """
    gets points, labels, features and sample weight tensors from dataset

    :param data_set: tf dataset to load from
    :param sess: tf session
    :param batch_size: batch size
    :param color: include colors in features
    :param normal: include normals in features
    :return: points(BxNx3), labels(BxN), features(BxNx?), sample_weigth(BxN)
    """
    data = data_set.batch(batch_size).prefetch(4)
    iterator = tf.data.Iterator.from_structure(data.output_types, data.output_shapes)
    data_init = iterator.make_initializer(data)
    sess.run(data_init)
    points, labels, colors, normals, sample_weight = iterator.get_next()
    colors = tf.div(tf.cast(colors, tf.float32), tf.constant(255, dtype=tf.float32))

    if color and normal:
        features = tf.concat([colors, normals], 2)
    elif color:
        features = colors
    elif normal:
        features = normals
    else:
        features = None

    mask = tf.not_equal(sample_weight, 0.0)
    mask = tf.cast(mask, tf.float32)
    sample_weight = tf.multiply(tf.gather(CLASS_WEIGHTS, labels), mask)
    return points, labels, features, sample_weight


def get_metrics(get_model: Callable,
                coordinates: tf.Tensor,
                features: Optional[tf.Tensor],
                is_training: tf.Variable,
                bn_decay: tf.Variable,
                labels: tf.Tensor,
                sample_weight: tf.Tensor,
                train: bool) \
        -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Operation, tf.Tensor, tf.Operation]:
    """
    gets model metrics

    :param get_model: method which gives prediction tensor back
    :param coordinates: point coordinates (BxNx3)
    :param features: feature tensor (optional) (BxNx?)
    :param is_training: tf var which indicates if model is training
    :param bn_decay: tf var which gives batch norm decay
    :param labels: label tensor (BxN)
    :param sample_weight: sample weight tensor (BxN)
    :param train: bool, which indicates whether these are train metrics
    :return: loss, accuracy, predictions(BxNxC), iou_update operation, iou, iou_reset operation
    """
    if features is not None:
        pred, _ = get_model(coordinates, features, is_training, 21, bn_decay=bn_decay)
    else:
        pred, _ = get_model(coordinates, is_training, 21, bn_decay=bn_decay)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=pred, weights=sample_weight)

    # Filter out the unassigned labels
    labels_flat = tf.reshape(labels, [-1])
    pred_flat = tf.reshape(pred, [-1, pred.shape[2]])
    loc = tf.reshape(tf.where(labels_flat > 0), [-1])
    labels_assigned = tf.gather(labels_flat, loc, axis=0)
    pred_assigned = tf.gather(pred_flat, loc)
    correct_pred = tf.equal(tf.argmax(pred_assigned, 1, output_type=tf.int32), labels_assigned)

    if train:
        prefix = "train"
    else:
        prefix = "val"

    iou, iou_update = tf.metrics.mean_iou(labels_assigned, tf.argmax(pred_assigned, 1, output_type=tf.int32),
                                          num_classes=21, name=prefix + "_iou")
    running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope=prefix + "_iou")
    iou_reset = tf.variables_initializer(var_list=running_vars)
    acc = tf.reduce_sum(tf.cast(correct_pred, tf.float32)) / \
          tf.cast(tf.shape(labels_assigned)[0], tf.float32)
    return loss, acc, pred, iou_update, iou, iou_reset


def get_tf_summary(loss: float, acc: float, iou: float) -> tf.Summary:
    """
    creates a tf Summary with loss, accuracy and iou

    :param loss: loss value
    :param acc: accuracy value
    :param iou: iou value
    :return: summary
    """
    summary = tf.Summary()
    summary.value.add(tag="loss", simple_value=loss)
    summary.value.add(tag="accuracy", simple_value=acc)
    summary.value.add(tag="iou", simple_value=iou)
    return summary


def summarize_epoch(epoch: int,
                    sess: tf.Session,
                    learning_rate: tf.Variable,
                    bn_decay: tf.Variable,
                    loss_sum: float,
                    batches_per_epoch: float,
                    acc_sum: float,
                    train_iou_val: float,
                    train_writer: tf.summary.FileWriter,
                    train_iou_reset: tf.Operation):
    """
    summarizes train metrics of one epoch

    :param epoch: index of epoch
    :param sess: tf session
    :param learning_rate:
    :param bn_decay:
    :param loss_sum: accumulated loss
    :param batches_per_epoch: number of batches used in an epoch
    :param acc_sum: accumulated accuracy
    :param train_iou_val: accumulated train iou
    :param train_writer: train summary writer
    :param train_iou_reset: operation to reset train iou
    :return:
    """
    lr, bn_d = sess.run([learning_rate, bn_decay])
    epoch_loss = loss_sum / batches_per_epoch
    epoch_acc = acc_sum / batches_per_epoch
    epoch_iou = train_iou_val
    print(f"mean loss: {epoch_loss:.4f}\tmean acc: {epoch_acc:.4f}\tmean iou: {epoch_iou:.4f}")
    summary = get_tf_summary(epoch_loss, epoch_acc, epoch_iou)
    summary.value.add(tag="learning_rate", simple_value=lr)
    summary.value.add(tag="bn_decay", simple_value=bn_d)
    train_writer.add_summary(summary, epoch)
    # reset accumulator
    sess.run(train_iou_reset)


def eval_model(is_training: tf.Variable,
               sess: tf.Session,
               best_iou: float,
               val_loss: tf.Tensor,
               val_acc: tf.Tensor,
               val_iou_update: tf.Operation,
               val_iou: tf.Tensor,
               val_iou_reset: tf.Operation,
               val_writer: tf.summary.FileWriter,
               epoch: int,
               saver: tf.train.Saver) -> float:
    """
    evaluates model with one pass over validation set

    :param is_training: tf var which indicates if model is training
    :param sess: tf sess
    :param best_iou: best validation iou until now
    :param val_loss: val loss tensor
    :param val_acc: val accuracy tensor
    :param val_iou_update: val iou update operation
    :param val_iou: val iou tensor
    :param val_iou_reset: val iou reset operation
    :param val_writer: val summary writer
    :param epoch: index of current epoch
    :param saver: tf model saver
    :return: new best iou
    """
    acc_sum, loss_sum = 0, 0

    # toggle training off
    assign_op = is_training.assign(False)
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
    loss = loss_sum / val_batches
    acc = acc_sum / val_batches
    iou = val_iou_val
    summary = get_tf_summary(loss, acc, iou)
    val_writer.add_summary(summary, epoch)
    print(f"evaluation:\tmean loss: {loss:.4f}\tmean acc: {acc:.4f}\tmean iou {iou:.4f}\n")

    # save model if it is better
    if iou > best_iou:
        best_iou = iou
        save_path = saver.save(sess, os.path.join(LOG_DIR + "_train", f"best_model_epoch_{epoch:03d}.ckpt"))
        print(f"Model saved in file: {save_path}\n")

    # reset accumulator
    sess.run(val_iou_reset)

    # toggle training on
    assign_op = is_training.assign(True)
    sess.run(assign_op)

    return best_iou


def train(epochs=1000, batch_size=BATCH_SIZE, use_color: bool = True, use_normal: bool = True, n_epochs_to_val=4):
    tf.Graph().as_default()
    tf.device('/gpu:0')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    # define train data
    train_data = precompute_dataset.get_precomputed_train_data_set()
    train_coordinates, train_labels, train_features, train_sample_weight = \
        get_data_tensors(train_data, sess, batch_size, use_color, use_normal)

    # define validation data
    val_data = precompute_dataset.get_precomputed_val_data_set()
    val_coordinates, val_labels, val_features, val_sample_weight = \
        get_data_tensors(val_data, sess, batch_size, use_color, use_normal)

    # define model and metrics
    is_training = tf.Variable(True)
    step = tf.Variable(0, trainable=False)
    bn_decay = get_bn_decay(step)
    learning_rate = get_learning_rate(step)

    if use_normal or use_color:
        model = pointnet2_sem_seg_features
    else:
        model = pointnet2_sem_seg
    # TODO model

    # train metrics
    train_loss, train_acc, train_pred, train_iou_update, train_iou, train_iou_reset = \
        get_metrics(model.get_model, train_coordinates, train_features, is_training, bn_decay, train_labels,
                    train_sample_weight, True)
    optimizer = tf.train.AdamOptimizer(learning_rate)

    train_op = optimizer.minimize(train_loss, global_step=step)

    # validation metrics
    val_loss, val_acc, val_pred, val_iou_update, val_iou, val_iou_reset = \
        get_metrics(model.get_model, val_coordinates, val_features, is_training, bn_decay, val_labels,
                    train_sample_weight, False)

    # initialize variables
    variable_init = tf.global_variables_initializer()
    sess.run(variable_init)
    sess.run(tf.local_variables_initializer())
    assign_op = is_training.assign(True)
    sess.run(assign_op)

    # add summaries
    train_writer = tf.summary.FileWriter(LOG_DIR + "_train", sess.graph)
    val_writer = tf.summary.FileWriter(LOG_DIR + "_val")
    saver = tf.train.Saver()

    batches_per_epoch = N_TRAIN_SAMPLES / batch_size
    print(f"batches per epoch: {batches_per_epoch}")

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

        if (i + 1) % int(batches_per_epoch) == 0:
            # end of epoch
            print(f"epoch {epoch} finished")
            summarize_epoch(epoch, sess, learning_rate, bn_decay, loss_sum, batches_per_epoch,
                            acc_sum, train_iou_val, train_writer, train_iou_reset)
            acc_sum, loss_sum = 0, 0
            if epoch % n_epochs_to_val == 0:
                # pass over validation set
                best_iou = eval_model(is_training, sess, best_iou, val_loss, val_acc, val_iou_update, val_iou,
                                      val_iou_reset, val_writer, epoch, saver)
            print(f"starting epoch {epoch + 1}")


if __name__ == '__main__':
    train()

import os
import sys
from datetime import datetime

import numpy as np
import tensorflow as tf

from attention_points.attention_scannet import scannet_dataset
from attention_points.attention_scannet.attention_models import AttentionNetModel
from pointnet2_tensorflow.scannet import pc_util
from pointnet2_tensorflow.utils import provider

GPU_INDEX = 0
BATCH_SIZE = 1

NUM_POINT = 8192
NUM_CLASSES = 21

OPTIMIZER = 'adam'
MOMENTUM = 0.9
BASE_LEARNING_RATE = 0.002
DECAY_STEP = 200000
DECAY_RATE = 0.7

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

LOG_DIR = os.path.join('/tmp/pycharm_project_250/pointnet2_tensorflow/log/%s_lr=2e-3' % datetime.now().isoformat())
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
MAX_EPOCH = 5000
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')

DATA_PATH = "/home/tim/data/"
TRAIN_DATASET = scannet_dataset.ScannetDataset(root=DATA_PATH, npoints=NUM_POINT, split='train')
TEST_DATASET = TRAIN_DATASET  # TODO this works only for overfit !!

EPOCH_CNT = 0


def get_batch_wdp(dataset, idxs, start_idx, end_idx):
    bsize = end_idx - start_idx
    batch_data = np.zeros((bsize, NUM_POINT, 3))
    batch_label = np.zeros((bsize, NUM_POINT), dtype=np.int32)
    batch_smpw = np.zeros((bsize, NUM_POINT), dtype=np.float32)
    for i in range(bsize):
        ps, seg, smpw = dataset[idxs[i + start_idx]]
        batch_data[i, ...] = ps
        batch_label[i, :] = seg
        batch_smpw[i, :] = smpw

        dropout_ratio = np.random.random() * 0.875  # 0-0.875
        drop_idx = np.where(np.random.random((ps.shape[0])) <= dropout_ratio)[0]
        batch_data[i, drop_idx, :] = batch_data[i, 0, :]
        batch_label[i, drop_idx] = batch_label[i, 0]
        batch_smpw[i, drop_idx] *= 0
    return batch_data, batch_label, batch_smpw


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
        BN_INIT_DECAY,
        batch * BATCH_SIZE,
        BN_DECAY_DECAY_STEP,
        BN_DECAY_DECAY_RATE,
        staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay


def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:' + str(GPU_INDEX)):
            is_training_pl = tf.placeholder(tf.bool, shape=(), name="is_training")
            model = AttentionNetModel(is_training=is_training_pl, bn_decay=None, num_class=NUM_CLASSES)

            pointclouds_pl, labels_pl, smpws_pl = model.placeholder_inputs(BATCH_SIZE, NUM_POINT)

            print("label shape: ", labels_pl.shape)

            print("--- Get model and loss")
            pred = model(pointclouds_pl)
            print("prediction shape", pred.shape)
            loss = tf.losses.sparse_softmax_cross_entropy(labels=labels_pl, logits=pred, weights=smpws_pl)
            # Get model and loss
            tf.summary.scalar('loss', loss)

            correct = tf.equal(tf.argmax(pred, 2), tf.to_int64(labels_pl))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE * NUM_POINT)
            tf.summary.scalar('accuracy', accuracy)

            print("--- Get training operator")
            # Get training operator
            batch = tf.Variable(0)
            learning_rate = tf.Variable(BASE_LEARNING_RATE)
            tf.summary.scalar('learning_rate', learning_rate)
            optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)

            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'), sess.graph)

        # Init variables
        init = tf.global_variables_initializer()
        sess.run(init)
        # sess.run(init, {is_training_pl: True})

        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'smpws_pl': smpws_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch}

        # get number of model parameters
        print(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
        best_acc = -1
        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()

            train_one_epoch(sess, ops, train_writer)
            acc = eval_one_epoch(sess, ops, test_writer)
            if epoch % 1000 == 0:
                learning_rate = learning_rate / 2
            # TODO implement the following
            # if epoch % 5 == 0:
            #   acc = eval_one_epoch(sess, ops, test_writer)
            #   acc = eval_whole_scene_one_epoch(sess, ops, test_writer)
            # if acc > best_acc:
            #     best_acc = acc
            #     save_path = saver.save(sess, os.path.join(LOG_DIR, "best_model_epoch_%03d.ckpt" % (epoch)))
            #     log_string("Model saved in file: %s" % save_path)
            #
            # # Save the variables to disk.
            # if epoch % 10 == 0:
            #     save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
            #     log_string("Model saved in file: %s" % save_path)


def train_one_epoch(sess, ops, train_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = True

    # Shuffle train samples
    train_idxs = np.arange(0, len(TRAIN_DATASET))
    np.random.shuffle(train_idxs)
    num_batches = len(TRAIN_DATASET) // BATCH_SIZE

    log_string(str(datetime.now()))

    total_correct = 0
    total_seen = 0
    loss_sum = 0
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx + 1) * BATCH_SIZE
        batch_data, batch_label, batch_smpw = get_batch_wdp(TRAIN_DATASET, train_idxs, start_idx, end_idx)
        # Augment batched point clouds by rotation
        aug_data = provider.rotate_point_cloud_z(batch_data)
        feed_dict = {ops['pointclouds_pl']: aug_data,
                     ops['labels_pl']: batch_label,
                     ops['smpws_pl']: batch_smpw,
                     ops['is_training_pl']: is_training, }
        summary, step, _, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
                                                         ops['train_op'], ops['loss'], ops['pred']],
                                                        feed_dict=feed_dict)
        train_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 2)
        correct = np.sum(pred_val == batch_label)
        total_correct += correct
        total_seen += (BATCH_SIZE * NUM_POINT)
        loss_sum += loss_val
        if (batch_idx + 1) % 10 == 0:
            log_string(' -- %03d / %03d --' % (batch_idx + 1, num_batches))
            log_string('mean loss: %f' % (loss_sum / 10))
            log_string('accuracy: %f' % (total_correct / float(total_seen)))
            total_correct = 0
            total_seen = 0
            loss_sum = 0


def eval_one_epoch(sess, ops, test_writer):
    """ ops: dict mapping from string to tf ops """
    global EPOCH_CNT
    is_training = False
    test_idxs = np.arange(0, len(TEST_DATASET))
    num_batches = len(TEST_DATASET) // BATCH_SIZE

    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]

    total_correct_vox = 0
    total_seen_vox = 0
    total_seen_class_vox = [0 for _ in range(NUM_CLASSES)]
    total_correct_class_vox = [0 for _ in range(NUM_CLASSES)]

    log_string(str(datetime.now()))
    log_string('---- EPOCH %03d EVALUATION ----' % EPOCH_CNT)

    labelweights = np.zeros(21)
    labelweights_vox = np.zeros(21)
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx + 1) * BATCH_SIZE
        batch_data, batch_label, batch_smpw = get_batch(TEST_DATASET, test_idxs, start_idx, end_idx)

        aug_data = provider.rotate_point_cloud_z(batch_data)

        feed_dict = {ops['pointclouds_pl']: aug_data,
                     ops['labels_pl']: batch_label,
                     ops['smpws_pl']: batch_smpw,
                     ops['is_training_pl']: is_training}
        summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
                                                      ops['loss'], ops['pred']], feed_dict=feed_dict)
        test_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 2)  # BxN
        correct = np.sum((pred_val == batch_label) & (batch_label > 0) & (
                batch_smpw > 0))  # evaluate only on 20 categories but not unknown
        total_correct += correct
        total_seen += np.sum((batch_label > 0) & (batch_smpw > 0))
        loss_sum += loss_val
        tmp, _ = np.histogram(batch_label, range(22))
        labelweights += tmp
        for l in range(NUM_CLASSES):
            total_seen_class[l] += np.sum((batch_label == l) & (batch_smpw > 0))
            total_correct_class[l] += np.sum((pred_val == l) & (batch_label == l) & (batch_smpw > 0))

        for b in range(batch_label.shape[0]):
            _, uvlabel, _ = \
                pc_util.point_cloud_label_to_surface_voxel_label_fast(
                    aug_data[b, batch_smpw[b, :] > 0, :], np.concatenate((np.expand_dims(
                        batch_label[b, batch_smpw[b, :] > 0], 1), np.expand_dims(pred_val[b, batch_smpw[b, :] > 0], 1)),
                                                                         axis=1), res=0.02)
            total_correct_vox += np.sum((uvlabel[:, 0] == uvlabel[:, 1]) & (uvlabel[:, 0] > 0))
            total_seen_vox += np.sum(uvlabel[:, 0] > 0)
            tmp, _ = np.histogram(uvlabel[:, 0], range(22))
            labelweights_vox += tmp
            for l in range(NUM_CLASSES):
                total_seen_class_vox[l] += np.sum(uvlabel[:, 0] == l)
                total_correct_class_vox[l] += np.sum((uvlabel[:, 0] == l) & (uvlabel[:, 1] == l))

    log_string('eval mean loss: %f' % (loss_sum / float(num_batches)))
    log_string('eval point accuracy vox: %f' % (total_correct_vox / float(total_seen_vox)))
    log_string('eval point avg class acc vox: %f' % (
        np.mean(np.array(total_correct_class_vox[1:]) / (np.array(total_seen_class_vox[1:], dtype=np.float) + 1e-6))))
    log_string('eval point accuracy: %f' % (total_correct / float(total_seen)))
    log_string('eval point avg class acc: %f' % (
        np.mean(np.array(total_correct_class[1:]) / (np.array(total_seen_class[1:], dtype=np.float) + 1e-6))))
    labelweights_vox = labelweights_vox[1:].astype(np.float32) / np.sum(labelweights_vox[1:].astype(np.float32))
    caliweights = np.array(
        [0.388, 0.357, 0.038, 0.033, 0.017, 0.02, 0.016, 0.025, 0.002, 0.002, 0.002, 0.007, 0.006, 0.022, 0.004, 0.0004,
         0.003, 0.002, 0.024, 0.029])
    log_string('eval point calibrated average acc: %f' % (
        np.average(np.array(total_correct_class[1:]) / (np.array(total_seen_class[1:], dtype=np.float) + 1e-6),
                   weights=caliweights)))
    per_class_str = 'vox based --------'
    for l in range(1, NUM_CLASSES):
        per_class_str += 'class %d weight: %f, acc: %f; ' % (
            l, labelweights_vox[l - 1], total_correct_class[l] / float(total_seen_class[l]))
    log_string(per_class_str)
    EPOCH_CNT += 1
    return total_correct / float(total_seen)


def get_batch(dataset, idxs, start_idx, end_idx):
    bsize = end_idx - start_idx
    batch_data = np.zeros((bsize, NUM_POINT, 3))
    batch_label = np.zeros((bsize, NUM_POINT), dtype=np.int32)
    batch_smpw = np.zeros((bsize, NUM_POINT), dtype=np.float32)
    for i in range(bsize):
        ps, seg, smpw = dataset[idxs[i + start_idx]]
        batch_data[i, ...] = ps
        batch_label[i, :] = seg
        batch_smpw[i, :] = smpw
    return batch_data, batch_label, batch_smpw


if __name__ == '__main__':
    train()

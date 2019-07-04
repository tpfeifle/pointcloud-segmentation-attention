import tensorflow as tf

from attention_scannet.attention_models import AttentionNetModel
from scannet_dataset import data_transformation
import importlib
import pickle

N_POINTS = 8192
N_TRAIN_SAMPLES = 1201
N_VAL_SAMPLES = 312
MODEL = importlib.import_module("models.pointnet2_sem_seg")


def train(epochs=1000, batch_size=8, n_epochs_to_val=4):
    tf.Graph().as_default()
    tf.device('/gpu:0')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    # define train
    train_data = data_transformation.get_transformed_dataset("train")
    train_data = train_data.batch(batch_size).prefetch(2)
    train_iterator = tf.data.Iterator.from_structure(train_data.output_types, train_data.output_shapes)
    train_data_init = train_iterator.make_initializer(train_data)
    sess.run(train_data_init)
    points, labels, colors, normals, sample_weight = train_iterator.get_next()
    train_features = tf.concat([tf.cast(colors, tf.float32), normals], 0)
    train_coordinates = points
    train_labels = labels
    train_sample_weight = sample_weight
    # define validation data
    val_data = data_transformation.get_transformed_dataset("val")
    val_data = val_data.prefetch(2)
    val_iterator = tf.data.Iterator.from_structure(val_data.output_types, train_data.output_shapes)
    val_data_init = val_iterator.make_initializer(val_data)
    sess.run(val_data_init)
    points, labels, colors, normals, sample_weight = train_iterator.get_next()
    val_features = tf.concat([tf.cast(colors, tf.float32), normals], 0)
    val_coordinates = points
    val_labels = labels
    val_sample_weight = sample_weight

    # define model and metrics
    # is_training_pl = tf.constant(True, tf.bool, shape=(), name="is_training")
    is_training_pl = tf.Variable(True)
    # model = AttentionNetModel(is_training=is_training_pl, bn_decay=None, num_class=21)

    # train_pred = model(train_coordinates)
    train_pred, _ = MODEL.get_model(train_coordinates, is_training_pl, 21)
    train_loss = tf.losses.sparse_softmax_cross_entropy(labels=train_labels, logits=train_pred,
                                                        weights=train_sample_weight)

    correct_train_pred = tf.equal(tf.argmax(train_pred, 2, output_type=tf.int32), train_labels)
    train_acc = tf.reduce_sum(tf.cast(correct_train_pred, tf.float32)) / float(batch_size * N_POINTS)
    optimizer = tf.train.AdamOptimizer(1e-3)
    train_op = optimizer.minimize(train_loss)
    # val_pred = model(val_coordinates)
    '''val_pred, _ = MODEL.get_model(val_coordinates, is_training_pl, 21)
    val_loss = tf.losses.sparse_softmax_cross_entropy(labels=val_labels, logits=val_pred,
                                                      weights=val_sample_weight)
    correct_val_pred = tf.equal(tf.argmax(val_pred, 2, output_type=tf.int32), val_labels)
    val_acc = tf.reduce_sum(tf.cast(correct_val_pred, tf.float32)) / \
              tf.cast(tf.shape(labels)[0] * N_POINTS, dtype=tf.float32)'''

    variable_init = tf.global_variables_initializer()
    sess.run(variable_init)

    batches_per_epoch = N_TRAIN_SAMPLES / batch_size
    # batches_per_epoch = 2
    print(f"batches per epoch: {batches_per_epoch}")
    acc_sum, loss_sum = 0, 0
    assign_op = is_training_pl.assign(True)
    sess.run(assign_op)
    print(tf.trainable_variables())
    for i in range(int(epochs * batches_per_epoch)):
        epoch = int((i + 1) / batches_per_epoch) + 1
        _, loss_val, acc_val, pred, batch_data = sess.run([train_op, train_loss, train_acc, train_pred, points])
        import numpy as np
        pred = np.argmax(pred, 2)
        acc_sum += acc_val
        loss_sum += loss_val
        print(f"\tbatch {(i + 1) % int(batches_per_epoch)}\tloss: {loss_val}, \taccuracy: {acc_val}")
        if (i + 1) % int(batches_per_epoch) == 0 or True:
            outfile = '/tmp/pycharm_project_250/to_visualize_pointnet.pickle'
            with open(outfile, 'wb') as fp:
                pickle.dump(batch_data, fp)
                pickle.dump(pred, fp)
            '''
            print(f"epoch {epoch} finished")
            # epoch summary
            print(f"mean acc: {acc_sum / batches_per_epoch} \tmean loss: {loss_sum / batches_per_epoch}")
            acc_sum, loss_sum = 0, 0
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

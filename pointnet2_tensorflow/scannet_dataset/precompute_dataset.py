import os.path
import pickle

import numpy as np
import tensorflow as tf

from scannet_dataset import data_transformation


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def deserialize_feature(feature):
    feature = {"points": tf.parse_tensor(feature["points"], np.float32),
               "labels": tf.parse_tensor(feature["labels"], np.int32),
               "colors": tf.parse_tensor(feature["colors"], np.int32),
               "normals": tf.parse_tensor(feature["normals"], np.float32),
               "sample_weights": tf.parse_tensor(feature["sample_weights"], np.float32)}
    return feature


def precompute_train_data(epochs, elements_per_epoch, out_dir, dataset, add_epoch=0):
    sess = tf.Session()
    data_iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
    train_data_init = data_iterator.make_initializer(dataset)
    sess.run(train_data_init)
    points, labels, colors, normals, sample_weight = data_iterator.get_next()
    for i in range(epochs):
        for j in range(elements_per_epoch):
            points_val, labels_val, colors_val, normals_val, sample_weight_val = sess.run(
                [points, labels, colors, normals, sample_weight])
            filename = f"{out_dir}/{i + add_epoch:03d}-{j:04d}.pickle"
            if not os.path.isfile(filename):
                with open(filename, "wb")as file:
                    pickle.dump((points_val, labels_val, colors_val, normals_val, sample_weight_val), file)
            else:
                raise ValueError("the file already exists")


def precomputed_data_generator(dir="/home/tim/data/train_precomputed"):
    file_list = sorted(os.listdir(dir))
    while True:
        for filename in file_list:
            if filename.endswith(".pickle"):
                file = (os.path.join(dir, filename))
                print(file)
                with open(file, "rb") as file:
                    points_val, labels_val, colors_val, normals_val, sample_weight_val = pickle.load(file)
                    yield points_val, labels_val, colors_val, normals_val, sample_weight_val


def get_precomputed_data_set():
    gen = precomputed_data_generator
    return tf.data.Dataset.from_generator(gen,
                                          output_types=(tf.float32, tf.int32, tf.int32, tf.float32, tf.float32),
                                          output_shapes=(tf.TensorShape([None, 3]), tf.TensorShape([None]),
                                                         tf.TensorShape([None, 3]), tf.TensorShape([None, 3]),
                                                         tf.TensorShape([None])))


def main():
    # ds = data_transformation.get_transformed_dataset("train").prefetch(4)
    # precompute_train_data(60, 1201, "/home/tim/data/train_precomputed", ds, 40)
    new_ds = get_precomputed_data_set()
    new_ds = new_ds.batch(16).prefetch(2)
    batch = new_ds.make_one_shot_iterator().get_next()
    sess = tf.Session()
    first_batch = sess.run(batch)
    print(first_batch)

if __name__ == '__main__':
    main()

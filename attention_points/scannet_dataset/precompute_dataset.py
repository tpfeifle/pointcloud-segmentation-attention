"""
This module provides methods to precompute, save and load training data.
This allows to speed up training, as the random subscenes do not have to be generated on the fly.
"""
import os.path
import pickle

import numpy as np
import tensorflow as tf

from attention_points.scannet_dataset import generator_dataset, data_transformation


def precompute_train_data(epochs, elements_per_epoch, out_dir, dataset, add_epoch=0):
    """
    precomputes train data
    :param epochs: number of epochs to precompute (random chunks do not cover whole scenes, set >=20)
    :param elements_per_epoch: number of elements loaded from dataset for a single epoch
    :param out_dir: directory to save files to
    :param dataset: tensorflow dataset to load scenes from
    :param add_epoch: offset of epoch name (to continue interrupted precomputation)
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
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


def precompute_val_data(elements, out_dir,
                        dataset=generator_dataset.get_dataset("val").prefetch(4).map(data_transformation.label_map)):
    """
    precomputes validation data
    :param elements: number of elements to load from dataset
    :param out_dir: directory to save files to
    :param dataset: tensorflow dataset to load scenes from
    """
    sess = tf.Session()
    data_iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
    val_data_init = data_iterator.make_initializer(dataset)
    sess.run(val_data_init)
    scene = data_iterator.get_next()
    for i in range(elements):
        scene_val = sess.run([scene])[0]
        points_val, labels_val, colors_val, normals_val = scene_val
        subscenes = data_transformation.get_all_subsets_for_scene_numpy(points_val, labels_val, colors_val, normals_val)
        for j in range(len(subscenes[0])):
            points_val, labels_val, colors_val, normals_val, sample_weight_val = (x[j] for x in subscenes)
            filename = f"{out_dir}/{i:03d}-{j:04d}.pickle"
            if not os.path.isfile(filename):
                with open(filename, "wb")as file:
                    pickle.dump((points_val, labels_val, colors_val, normals_val, sample_weight_val), file)
            else:
                raise ValueError("the file already exists")


def generate_eval_data():
    """
    generator which iterates over precomputed validation set and yields single subsets
    """
    for scene_name in generator_dataset.scene_name_generator("val"):
        print(scene_name)
        points_val, labels_val, colors_val, normals_val = generator_dataset.load_from_scene_name(scene_name)
        labels_val = data_transformation.label_map_more_paraemters(labels_val.astype(np.int32))
        subscenes = data_transformation.get_all_subsets_with_all_points_for_scene_numpy(points_val, labels_val,
                                                                                        colors_val, normals_val)

        for j in range(len(subscenes[0])):
            points_val, labels_val, colors_val, normals_val, sample_weight_val, mask, points_orig_idxs = (x[j] for x in
                                                                                                          subscenes)
            yield (points_val, labels_val, colors_val, normals_val, scene_name.encode('utf-8'), mask, points_orig_idxs)


def generate_test_data():
    """
    generator which iterates over precomputed test set and yields single subsets
    """
    for scene_name in generator_dataset.scene_name_generator("test"):
        points_val, colors_val, normals_val = generator_dataset.load_from_scene_name_test(scene_name)
        subscenes = data_transformation.get_all_subsets_with_all_points_for_scene_numpy_test(points_val, colors_val,
                                                                                             normals_val)

        for j in range(len(subscenes[0])):
            points_val, colors_val, normals_val, mask, points_orig_idxs = (x[j] for x in subscenes)
            yield (points_val, colors_val, normals_val, scene_name.encode('utf-8'), mask, points_orig_idxs)


def eval_dataset_from_generator():
    """
    dataset from eval data generator
    """
    gen = generate_eval_data
    return tf.data.Dataset.from_generator(gen,
                                          output_types=(tf.float32, tf.int32, tf.int32, tf.float32, tf.string, tf.int32,
                                                        tf.int32),
                                          output_shapes=(tf.TensorShape([None, 3]), tf.TensorShape([None]),
                                                         tf.TensorShape([None, 3]), tf.TensorShape([None, 3]),
                                                         tf.TensorShape([]), tf.TensorShape([None]),
                                                         tf.TensorShape([None])))


def test_dataset_from_generator():
    """
    tensorflow dataset from test data generator
    """
    gen = generate_test_data
    return tf.data.Dataset.from_generator(gen,
                                          output_types=(tf.float32, tf.int32, tf.float32, tf.string, tf.int32,
                                                        tf.int32),
                                          output_shapes=(tf.TensorShape([None, 3]),
                                                         tf.TensorShape([None, 3]), tf.TensorShape([None, 3]),
                                                         tf.TensorShape([]), tf.TensorShape([None]),
                                                         tf.TensorShape([None])))


def precomputed_train_data_generator(dir="/home/tim/data/train_precomputed"):
    """
    iterates over precomputed train data and yields single chunks
    """
    file_list = sorted(os.listdir(dir))
    while True:
        for filename in file_list:
            if filename.endswith(".pickle"):
                file = (os.path.join(dir, filename))
                with open(file, "rb") as file:
                    points_val, labels_val, colors_val, normals_val, sample_weight_val = pickle.load(file)
                    yield points_val, labels_val, colors_val, normals_val, sample_weight_val


def get_precomputed_train_data_set():
    """
    tensorflow dataset from precomputed train data generator
    """
    gen = precomputed_train_data_generator
    return tf.data.Dataset.from_generator(gen,
                                          output_types=(tf.float32, tf.int32, tf.int32, tf.float32, tf.float32),
                                          output_shapes=(tf.TensorShape([None, 3]), tf.TensorShape([None]),
                                                         tf.TensorShape([None, 3]), tf.TensorShape([None, 3]),
                                                         tf.TensorShape([None])))


def precomputed_val_data_generator(dir="/home/tim/data/val_precomputed"):
    """
    iterates over precomputed val data and yields single chunks
    """
    file_list = sorted(os.listdir(dir))
    while True:
        for filename in file_list:
            if filename.endswith(".pickle"):
                file = (os.path.join(dir, filename))
                with open(file, "rb") as file:
                    points_val, labels_val, colors_val, normals_val, sample_weight_val = pickle.load(file)
                    yield points_val, labels_val, colors_val, normals_val, sample_weight_val


def get_precomputed_val_data_set():
    """
    tensorflow dataset from precomputed val data generator
    """
    gen = precomputed_val_data_generator
    return tf.data.Dataset.from_generator(gen,
                                          output_types=(tf.float32, tf.int32, tf.int32, tf.float32, tf.float32),
                                          output_shapes=(tf.TensorShape([None, 3]), tf.TensorShape([None]),
                                                         tf.TensorShape([None, 3]), tf.TensorShape([None, 3]),
                                                         tf.TensorShape([None])))


def precomputed_train_subset_data_generator(dir="/home/tim/data/train_subset_precomputed"):
    """
    iterates over precomputed subset of train data and yields single chunks
    """
    file_list = sorted(os.listdir(dir))
    while True:
        for filename in file_list:
            if filename.endswith(".pickle"):
                file = (os.path.join(dir, filename))
                with open(file, "rb") as file:
                    points_val, labels_val, colors_val, normals_val, sample_weight_val = pickle.load(file)
                    yield points_val, labels_val, colors_val, normals_val, sample_weight_val


def get_precomputed_train_subset_data_set():
    """
    tensorflow dataset from precomputed subset of train data generator
    """
    gen = precomputed_train_data_generator
    return tf.data.Dataset.from_generator(gen,
                                          output_types=(tf.float32, tf.int32, tf.int32, tf.float32, tf.float32),
                                          output_shapes=(tf.TensorShape([None, 3]), tf.TensorShape([None]),
                                                         tf.TensorShape([None, 3]), tf.TensorShape([None, 3]),
                                                         tf.TensorShape([None])))


def precomputed_val_subset_data_generator(dir="/home/tim/data/val_precomputed"):
    """
    iterates over precomputed subset of val data and yields single chunks
    """
    file_list = sorted(os.listdir(dir))
    file_list = file_list[:len(file_list) // 3]
    print("val files", len(file_list))
    while True:
        for filename in file_list:
            if filename.endswith(".pickle"):
                file = (os.path.join(dir, filename))
                with open(file, "rb") as file:
                    points_val, labels_val, colors_val, normals_val, sample_weight_val = pickle.load(file)
                    yield points_val, labels_val, colors_val, normals_val, sample_weight_val


def get_precomputed_val_subset_data_set():
    """
    tensorflow dataset from precomputed subset of val data generator
    """
    gen = precomputed_val_subset_data_generator
    return tf.data.Dataset.from_generator(gen,
                                          output_types=(tf.float32, tf.int32, tf.int32, tf.float32, tf.float32),
                                          output_shapes=(tf.TensorShape([None, 3]), tf.TensorShape([None]),
                                                         tf.TensorShape([None, 3]), tf.TensorShape([None, 3]),
                                                         tf.TensorShape([None])))


def precompute_subset_train_data():
    """
    precomputes data from a subset of the scenes in the training set
    :return:
    """
    ds = data_transformation.get_transformed_dataset("train_subset").prefetch(4)
    precompute_train_data(100, 1201 // 3, "/home/tim/data/train_subset_precomputed", ds, 0)
    print("done")


def main():
    # debug validation iterator
    sess = tf.Session()
    val_ds = get_precomputed_val_data_set()
    batch = val_ds.make_one_shot_iterator().get_next()
    first_batch = sess.run(batch)
    print("val batch", first_batch)
    for i in first_batch:
        print(i.shape)

    # debug train iterator
    train_ds = get_precomputed_train_data_set()
    train_ds = train_ds.batch(16).prefetch(2)
    batch = train_ds.make_one_shot_iterator().get_next()
    first_batch = sess.run(batch)
    print("train batch", first_batch)

    # for data in generate_eval_data():
    #    a = data
    # precompute a subset of the data
    # precompute_subset_train_data()

    # precompute val data
    # precompute_val_data(312, "/home/tim/data/val_precomputed")
    # print("done")

    # precompute train data
    # ds = data_transformation.get_transformed_dataset("train").prefetch(4)
    # precompute_train_data(60, 1201, "/home/tim/data/train_precomputed", ds, 40)


if __name__ == '__main__':
    main()

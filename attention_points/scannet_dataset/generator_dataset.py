"""
this module provides several generators which yield scenes of the dataset
these generators are used to create tensorflow datasets
"""
import random
from typing import Generator, List

import numpy as np
import tensorflow as tf


def scene_name_generator(train: str, base_dir: str = "/home/tim/.max_remote_deployment/",
                         overfit: bool = False) -> Generator:
    """
    yields the scene names of either train, val or test set

    :param train: one of {'train', 'val', 'test', 'train_subset'}
    :param base_dir: the directory, where the code is stored at
    :param overfit: if true, this data loader always returns the same scene
    :return: yields the scene names
    """
    if train == "train":
        with open(base_dir + "attention_points/scannet_dataset/splits/scannetv2_train.txt") as f:
            scenes = f.readlines()
    elif train == "val":
        with open(base_dir + "attention_points/scannet_dataset/splits/scannetv2_val.txt") as f:
            scenes = f.readlines()
    elif train == "test":
        with open(base_dir + "attention_points/scannet_dataset/splits/scannetv2_test.txt") as f:
            scenes = f.readlines()
    elif train == "train_subset":
        with open(base_dir + "attention_points/scannet_dataset/splits/scannetv2_train.txt") as f:
            scenes = f.readlines()
            scenes = scenes[:len(scenes) // 3]
    else:
        raise ValueError("train must be 'train', 'val' or 'test'")
    if overfit:
        scenes = scenes[:1]
    while True:
        if train == "train" or train == "train_subset":
            random.shuffle(scenes)
        for scene in scenes:
            yield (scene[:-1])


def load_from_scene_name(scene_name: str, pre_files_dir: str = "/home/tim/scannet-pre/") -> List[np.ndarray]:
    """
    loads points, labels, colors and normals from the preprocessed data

    :param scene_name: name of a scene, e.g. "scene0720_00"
    :param pre_files_dir: location of preprocessed files
    :return: [points (Nx3), labels (N), colors (Nx3), normals(Nx3)]
    """
    points = np.load(pre_files_dir + "points/" + scene_name + ".npy")
    labels = np.load(pre_files_dir + "labels/" + scene_name + ".npy")
    colors = np.load(pre_files_dir + "colors/" + scene_name + "_vh_clean_2.ply.npy")
    normals = np.load(pre_files_dir + "normals/" + scene_name + "_vh_clean_2.ply.npy")
    for i in [labels, colors, normals]:
        assert len(i) == len(points)
    return [points, labels, colors, normals]


def load_from_scene_name_test(scene_name: str, pre_files_dir: str = "/home/tim/scannet_test_data/") -> List[np.ndarray]:
    """
    loads points, colors and normals from the preprocessed data
    does not load labels, so it can be used for test data

    :param scene_name: name of a scene, e.g. "scene0720_00"
    :param pre_files_dir: location of preprocessed files
    :return: [points (Nx3), colors (Nx3), normals(Nx3)]
    """
    points = np.load(pre_files_dir + "points/" + scene_name + "_vh_clean_2.ply.npy")
    colors = np.load(pre_files_dir + "colors/" + scene_name + "_vh_clean_2.ply.npy")
    normals = np.load(pre_files_dir + "normals/" + scene_name + "_vh_clean_2.ply.npy")
    for i in [colors, normals]:
        assert len(i) == len(points)
    return [points, colors, normals]


def tf_train_generator() -> Generator:
    """
    yields train scenes

    :return: (points, labels, colors, normals)
    """
    for scene_name in scene_name_generator("train"):
        points, labels, colors, normals = load_from_scene_name(scene_name)
        yield (points, labels, colors, normals)


def tf_val_generator() -> Generator:
    """
    yields validation scenes

    :return: (points, labels, colors, normals)
    """
    for scene_name in scene_name_generator("val"):
        points, labels, colors, normals = load_from_scene_name(scene_name)
        yield (points, labels, colors, normals)


def tf_train_subset_generator() -> Generator:
    """
    yields subset scenes
    :return: (points, labels, colors, normals)
    """
    for scene_name in scene_name_generator("train_subset"):
        points, labels, colors, normals = load_from_scene_name(scene_name)
        yield (points, labels, colors, normals)


def tf_eval_generator() -> Generator:
    """
    yields validation scenes with scene name

    :return: (points, labels, colors, normals, scene_name)
    """
    for scene_name in scene_name_generator("val"):
        points, labels, colors, normals = load_from_scene_name(scene_name)
        yield (points, labels, colors, normals, scene_name)


def tf_test_generator() -> Generator:
    """
    yields test scenes with scene name

    :return: (points, colors, normals, scene_name)
    """
    for scene_name in scene_name_generator("test"):
        points, colors, normals = load_from_scene_name_test(scene_name)
        yield (points, colors, normals, scene_name)


def get_dataset(train: str) -> tf.data.Dataset:
    """
    returns a tensorflow dataset with the data specified in train

    :param train: can be ["train", "val", "eval", "test", "train_subset"]
    :return: the tensorflow dataset
    """
    if train == "train":
        gen = tf_train_generator
        return tf.data.Dataset.from_generator(gen,
                                              output_types=(tf.float32, tf.int32, tf.int32, tf.float32),
                                              output_shapes=(tf.TensorShape([None, 3]), tf.TensorShape([None]),
                                                             tf.TensorShape([None, 3]), tf.TensorShape([None, 3])))
    elif train == "val":
        gen = tf_val_generator
        return tf.data.Dataset.from_generator(gen,
                                              output_types=(tf.float32, tf.int32, tf.int32, tf.float32),
                                              output_shapes=(tf.TensorShape([None, 3]), tf.TensorShape([None]),
                                                             tf.TensorShape([None, 3]), tf.TensorShape([None, 3])))
    elif train == "eval":
        gen = tf_eval_generator
        return tf.data.Dataset.from_generator(gen,
                                              output_types=(tf.float32, tf.int32, tf.int32, tf.float32, tf.string),
                                              output_shapes=(tf.TensorShape([None, 3]), tf.TensorShape([None]),
                                                             tf.TensorShape([None, 3]), tf.TensorShape([None, 3]),
                                                             tf.TensorShape([])))
    elif train == "test":
        gen = tf_test_generator
        return tf.data.Dataset.from_generator(gen,
                                              output_types=(tf.float32, tf.int32, tf.float32, tf.string),
                                              output_shapes=(tf.TensorShape([None, 3]),
                                                             tf.TensorShape([None, 3]), tf.TensorShape([None, 3]),
                                                             tf.TensorShape([])))
    elif train == "train_subset":
        gen = tf_train_subset_generator
        return tf.data.Dataset.from_generator(gen,
                                              output_types=(tf.float32, tf.int32, tf.int32, tf.float32),
                                              output_shapes=(tf.TensorShape([None, 3]), tf.TensorShape([None]),
                                                             tf.TensorShape([None, 3]), tf.TensorShape([None, 3])))

    else:
        raise ValueError("use an element of {'train', 'val', 'eval', 'test', 'train_subset'} for train")


if __name__ == '__main__':
    for i in tf_train_generator():
        points, labels, colors, normals = i
    dataset = get_dataset("train")
    sess = tf.Session()
    value = sess.run(dataset.make_one_shot_iterator().get_next())
    print(value)
    for i in value:
        print(type(i))

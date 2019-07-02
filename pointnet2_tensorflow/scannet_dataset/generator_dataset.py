import random
from typing import Generator, List

import numpy as np
import tensorflow as tf


def scene_name_generator(train: str) -> Generator:
    """
    yields the scene names of either train, val or test set

    :param train: one of ["train", "val", "test"]
    :return:
    """
    if train == "train":
        with open("splits/scannetv2_train.txt") as f:
            scenes = f.readlines()
    elif train == "val":
        with open("splits/scannetv2_val.txt") as f:
            scenes = f.readlines()
    elif train == "test":
        with open("splits/scannetv2_val.txt") as f:
            scenes = f.readlines()
    else:
        raise ValueError("train must be 'train', 'val' or 'test'")
    random.shuffle(scenes)
    for scene in scenes:
        yield (scene[:-1])


def load_from_scene_name(scene_name, pre_files_dir="/home/tim/scannet-pre/") -> List[np.ndarray]:
    points = np.load(pre_files_dir + "points/" + scene_name + ".npy")
    labels = np.load(pre_files_dir + "labels/" + scene_name + ".npy")
    colors = np.load(pre_files_dir + "colors/" + scene_name + "_vh_clean_2.ply.npy")
    normals = np.load(pre_files_dir + "normals/" + scene_name + "_vh_clean_2.ply.npy")
    for i in [labels, colors, normals]:
        assert len(i) == len(points)
    return [points, labels, colors, normals]


def tf_train_generator() -> Generator:
    for scene_name in scene_name_generator("train"):
        points, labels, colors, normals = load_from_scene_name(scene_name)
        yield (points, labels, colors, normals)


def tf_val_generator() -> Generator:
    for scene_name in scene_name_generator("train"):
        points, labels, colors, normals = load_from_scene_name(scene_name)
        yield (points, labels, colors, normals)


def get_dataset(train):
    if train == "train":
        gen = tf_train_generator
    elif train == "val":
        gen = tf_val_generator
    else:
        raise ValueError("use 'train' or 'val' for train")
    return tf.data.Dataset.from_generator(gen,
                                          output_types=(tf.float32, tf.int32, tf.int32, tf.float32),
                                          output_shapes=(tf.TensorShape([None, 3]), tf.TensorShape([None]),
                                                         tf.TensorShape([None, 3]), tf.TensorShape([None, 3])))


if __name__ == '__main__':
    for i in tf_train_generator():
        points, labels, colors, normals = i
    dataset = get_dataset("train")
    sess = tf.Session()
    value = sess.run(dataset.make_one_shot_iterator().get_next())
    print(value)
    for i in value:
        print(type(i))

"""
provides methods to use ScanNet Dataset
"""
import os
from typing import Tuple

import numpy as np
import pptk
from plyfile import PlyData


def read_mesh_vertices(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    takes a ply file and extracts vertex coordinates and colors

    :param filename: path to ply file
    :return: points(Nx3), colors(Nx3)
    """
    assert os.path.isfile(filename)
    with open(filename, 'rb') as f:
        ply_data = PlyData.read(f)
        nr_vertices = ply_data['vertex'].count
        vertices = np.zeros(shape=[nr_vertices, 3], dtype=np.float32)
        vertices[:, 0] = ply_data['vertex'].data['x']
        vertices[:, 1] = ply_data['vertex'].data['y']
        vertices[:, 2] = ply_data['vertex'].data['z']
        colors = np.zeros(shape=[nr_vertices, 3], dtype=int)
        colors[:, 0] = ply_data['vertex'].data['red']
        colors[:, 1] = ply_data['vertex'].data['green']
        colors[:, 2] = ply_data['vertex'].data['blue']
    return vertices, colors


def read_label(filename: str) -> np.ndarray:
    """
    takes a ply file and returns labels

    :param filename: path to ply file
    :return: labels(N)
    """
    assert os.path.isfile(filename)
    with open(filename, 'rb') as f:
        ply_data = PlyData.read(f)
        labels = ply_data['vertex'].data['label']
    return np.array(labels)


def to_np_array(filename: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    takes ply file and extracts point coordinates, colors, and labels
    warning, this method only works on label files, which have not the original color but the labels color assigned
    -> use with care

    :param filename: path to ply file
    :return: vertices(Nx3), colors(Nx3), labels(N)
    """
    assert os.path.isfile(filename)
    with open(filename, 'rb') as f:
        ply_data = PlyData.read(f)
        nr_vertices = ply_data['vertex'].count
        vertices = np.zeros(shape=[nr_vertices, 3], dtype=np.float32)
        vertices[:, 0] = ply_data['vertex'].data['x']
        vertices[:, 1] = ply_data['vertex'].data['y']
        vertices[:, 2] = ply_data['vertex'].data['z']
        # next line works only for non label files
        colors = np.zeros(shape=[nr_vertices, 3], dtype=np.int16)
        colors[:, 0] = ply_data['vertex'].data['red']
        colors[:, 1] = ply_data['vertex'].data['green']
        colors[:, 2] = ply_data['vertex'].data['blue']
        # next line works only for label files
        labels = np.array(ply_data['vertex'].data['label'], dtype=np.int16)
    return vertices, colors, labels


def visualize(file: str = "C:/scannet/scene0000_00_vh_clean_2.ply"):
    """
    visualizes ply file with pptk

    :param file: path to file to visualize
    :return:
    """
    v1, c1 = read_mesh_vertices(file)
    # v1 = v1[:20000]
    # c1 = c1[:20000]
    print(v1.shape)
    print(c1 / 255.0)
    c1 = c1 / 255.0
    print(np.amin(c1))
    print(np.amax(c1))
    view = pptk.viewer(v1)
    view.attributes(c1)


def copy_data_to_numpy(source_dir: str = "C:/scannet", target_dir: str = "C:/scannet-pre/"):
    """
    iterates over a folder with scene and label ply files and extracts coordinates, color and labels for each scene
    saves the result in the target directory

    :param source_dir: directory containing ply files
    :param target_dir: directory to store numpy files at
    :return:
    """
    for subdir, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith("_vh_clean_2.labels.ply"):
                print(f"open file: {subdir + '/' + file}")
                l = read_label(subdir + '/' + file)
                np.save(target_dir + "labels/" + file, l)
            if file.endswith("_vh_clean_2.ply"):
                print(f"open file: {subdir + '/' + file}")
                v, c = read_mesh_vertices(subdir + '/' + file)
                np.save(target_dir + "points/" + file, v)
                np.save(target_dir + "colors/" + file, c)


def copy_test_data_to_numpy(source_dir: str = "C:/scannet", target_dir: str = "C:/scannet-pre/"):
    """
    iterates over a folder with test ply files and extracts point coordinates and color for each scene
    saves the result in the target directory

    :param source_dir: directory containing ply files
    :param target_dir: directory to store numpy files at
    :return:
    """
    for subdir, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith("_vh_clean_2.ply"):
                print(f"open file: {subdir + '/' + file}")
                v, c = read_mesh_vertices(subdir + '/' + file)
                np.save(target_dir + "colors/" + file, c)
                np.save(target_dir + "points/" + file, v)


def rename_files(dir: str):
    """
    renames files to omit unnecessary endings

    :param dir: directory of files to rename
    :return:
    """
    files = os.listdir(dir)
    for index, file in enumerate(files):
        print("renaming: " + file[:12] + ".npy")
        os.rename(os.path.join(dir, file), os.path.join(dir, file[:12] + ".npy"))


if __name__ == '__main__':
    copy_data_to_numpy()
    copy_test_data_to_numpy()

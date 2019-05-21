"""
provides methods to use ScanNet Dataset
"""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
import os, sys
import json
import numpy as np
from plyfile import PlyData, PlyElement
import pptk
import torch.utils.data as data
import torch


def read_mesh_vertices(filename):
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


def read_label(filename):
    assert os.path.isfile(filename)
    with open(filename, 'rb') as f:
        ply_data = PlyData.read(f)
        labels = ply_data['vertex'].data['label']
    return np.array(labels)


def to_np_array(filename):
    assert os.path.isfile(filename)
    with open(filename, 'rb') as f:
        ply_data = PlyData.read(f)
        nr_vertices = ply_data['vertex'].count
        vertices = np.zeros(shape=[nr_vertices, 3], dtype=np.float32)
        vertices[:, 0] = ply_data['vertex'].data['x']
        vertices[:, 1] = ply_data['vertex'].data['y']
        vertices[:, 2] = ply_data['vertex'].data['z']
        colors = np.zeros(shape=[nr_vertices, 3], dtype=np.int16)
        colors[:, 0] = ply_data['vertex'].data['red']
        colors[:, 1] = ply_data['vertex'].data['green']
        colors[:, 2] = ply_data['vertex'].data['blue']
        labels = np.array(ply_data['vertex'].data['label'], dtype=np.int16)
    return vertices, colors, labels


def visualize():
    v1, c1 = read_mesh_vertices("C:/scannet/scene0000_00_vh_clean_2.ply")
    # v1 = v1[:20000]
    # c1 = c1[:20000]
    print(v1.shape)
    print(c1 / 255.0)
    c1 = c1 / 255.0
    print(np.amin(c1))
    print(np.amax(c1))
    view = pptk.viewer(v1)
    view.attributes(c1)


def copy_data_to_numpy():
    target_dir = "C:/scannet-pre/"
    for subdir, dirs, files in os.walk("C:/scannet"):
        for file in files:
            if file.endswith("_vh_clean_2.labels.ply"):
                print(f"open file: {subdir + '/' + file}")
                v, c, l = to_np_array(subdir + '/' + file)
                np.save(target_dir + "points/" + file, v)
                np.save(target_dir + "colors/" + file, c)
                np.save(target_dir + "labels/" + file, l)


def rename_files(dir):
    files = os.listdir(dir)
    for index, file in enumerate(files):
        print("renaming: " + file[:12] + ".npy")
        os.rename(os.path.join(dir, file), os.path.join(dir, file[:12] + ".npy"))


def get_random_subset(arrays, nr):
    perm = np.random.permutation(len(arrays[0]))
    result = []
    for arr in arrays:
        arr = np.array([arr[i] for i in perm])
        result.append(arr[:nr])
    return result


def label_stats():
    dataset = ScanNetDataset()
    res = dict()
    for i in range(len(dataset)):
        sample = dataset[i]
        labels = sample["label"].numpy()
        for elem in labels:
            if elem in res:
                res[elem] += 1
            else:
                res[elem] = 1
        if i % 50 == 0:
            print(f"after {i} samples:")
            print(res)
    print("done")
    print(res)
    D = res
    plt.bar(range(len(D)), list(D.values()), align='center')
    plt.xticks(range(len(D)), list(D.keys()))
    plt.show()
    return res


class ScanNetDataset(data.Dataset):
    def __init__(self, train=True):
        if train:
            with open("splits/scannetv2_train.txt") as f:
                self.items = f.readlines()
        else:
            with open("splits/scannetv2_val.txt") as f:
                self.items = f.readlines()
        self.points = ["C:/scannet-pre/points/" + f[:-1] + ".npy" for f in self.items]
        self.colors = ["C:/scannet-pre/colors/" + f[:-1] + ".npy" for f in self.items]
        self.labels = ["C:/scannet-pre/labels/" + f[:-1] + ".npy" for f in self.items]
        for f in self.points:
            assert os.path.isfile(f)
        for f in self.colors:
            assert os.path.isfile(f)
        for f in self.labels:
            assert os.path.isfile(f)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        # load numpy arrays from disk
        point = np.load(self.points[idx])
        color = np.load(self.colors[idx])
        label = np.load(self.labels[idx]) - 1  # TODO labels go from 0 to 39

        # mask invalid labels
        mask = np.logical_and(label >= 0, label < 40)
        point = point[mask]
        color = color[mask]
        label = label[mask]

        # TODO remove subsampling in following line
        point, color, label = get_random_subset([point, color, label], 5000)

        # convert to torch tensors
        point = torch.from_numpy(point)
        color = torch.from_numpy(color)
        label = torch.from_numpy(label).type(torch.LongTensor)
        assert point.shape[0] == color.shape[0] == label.shape[0]
        sample = {'point': point, 'color': color, 'label': label}
        return sample


class ScanNetDatasetWrapper(ScanNetDataset):
    """
    A wrapper, which only loads points and labels, not color
    """

    def __getitem__(self, idx):
        sample = super(ScanNetDatasetWrapper, self).__getitem__(idx)
        return sample['point'], sample['label']


if __name__ == '__main__':
    # visualize()
    # print(read_label("C:/scannet/scene0000_00_vh_clean_2.labels.ply"))
    # print(read_label("C:/scannet/scene0000_00_vh_clean_2.labels.ply").shape)
    # copy_data_to_numpy()
    # rename_files("C:/scannet-pre/points")
    # rename_files("C:/scannet-pre/colors")
    # rename_files("C:/scannet-pre/labels")
    data = ScanNetDatasetWrapper(True)
    label_stats()

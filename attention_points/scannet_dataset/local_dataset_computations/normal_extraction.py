"""
provides methods to use ScanNet Dataset

Is used to copy normal vectors from .ply to .npy
"""
import os

import numpy as np
from plyfile import PlyData


def normals_to_np_array(filename):
    assert os.path.isfile(filename)
    with open(filename, 'rb') as f:
        ply_data = PlyData.read(f)
        nr_vertices = ply_data['vertex'].count
        normals = np.zeros(shape=[nr_vertices, 3], dtype=np.float32)
        normals[:, 0] = ply_data['vertex'].data['nx']
        normals[:, 1] = ply_data['vertex'].data['ny']
        normals[:, 2] = ply_data['vertex'].data['nz']
        return normals


def read_normal_vectors(source_dir="C:/scannet_normal/", target_dir="C:/scannet-pre/"):
    i = 0
    for subdir, dirs, files in os.walk(source_dir):
        for file in files:
            i += 1
            if file.endswith("_vh_clean_2.ply"):
                print(f"open file: {subdir + '/' + file}")
                n = normals_to_np_array(subdir + '/' + file)
                np.save(target_dir + "normals/" + file, n)


if __name__ == '__main__':
    read_normal_vectors()

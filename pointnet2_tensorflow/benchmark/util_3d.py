import os

import numpy as np
from plyfile import PlyData


# matrix: 4x4 np array
# points Nx3 np array
def transform_points(matrix, points):
    assert len(points.shape) == 2 and points.shape[1] == 3
    num_points = points.shape[0]
    p = np.concatenate([points, np.ones((num_points, 1))], axis=1)
    p = np.matmul(matrix, np.transpose(p))
    p = np.transpose(p)
    p[:, :3] /= p[:, 3, None]
    return p[:, :3]


def export_ids(filename, ids):
    with open(filename, 'w') as f:
        for id in ids:
            f.write('%d\n' % id)


def load_ids(filename):
    ids = open(filename).read().splitlines()
    ids = np.array(ids, dtype=np.int64)
    return ids


def read_mesh_vertices(filename):
    assert os.path.isfile(filename)
    with open(filename, 'rb') as f:
        plydata = PlyData.read(f)
        num_verts = plydata['vertex'].count
        vertices = np.zeros(shape=[num_verts, 3], dtype=np.float32)
        vertices[:, 0] = plydata['vertex'].data['x']
        vertices[:, 1] = plydata['vertex'].data['y']
        vertices[:, 2] = plydata['vertex'].data['z']
    return vertices
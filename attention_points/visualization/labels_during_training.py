"""
Animate the prediction changes in a scene over training time by reading the predictions at different time steps during
the training process

Input:

- Path to a pickle file containing the points of a scene
- List of paths to pickle files containing the labels at different time steps of the training process
"""

import os
import pickle
import time
from typing import List

import pptk

files = [os.path.join('/tmp/to_visualize1562169522_19.pickle'),
         os.path.join('/tmp/to_visualize1562169522_29.pickle'),
         os.path.join('/tmp/to_visualize1562169522_74.pickle')]

scene_points = os.path.join('/tmp/to_visualize1562169522_74.pickle')

g_label_names = ['unannotated', 'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
                 'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator', 'shower curtain', 'toilet',
                 'sink', 'bathtub', 'otherfurniture']

g_label_colors = [
    [0, 0, 0], [173, 198, 232], [151, 223, 137], [30, 120, 180], [255, 188, 119], [188, 189, 36], [140, 86, 74],
    [255, 152, 151], [213, 38, 40], [197, 175, 213], [148, 103, 188], [197, 156, 148], [24, 190, 208], [247, 183, 210],
    [218, 219, 141], [254, 127, 11], [158, 218, 229], [43, 160, 45], [111, 128, 144], [227, 120, 193], [82, 82, 163]
]


def normalize_colors():
    """
    Normalize colors to range [0, 1]: needed for pptk-viewer
    :return:
    """
    for idx, color_val in enumerate(g_label_colors):
        g_label_colors[idx] = [color_val[0] / 255.0, color_val[1] / 255.0, color_val[2] / 255.0]


def animate_prediction_changes(points_path: str, files_to_visualize: List[str]):
    """

    :param points_path: Path to pickle file containing the points of a scene
    :param files_to_visualize: List of paths to pickle files containing the labels at different time steps
                               of the training process
    :return:
    """
    # Read the points from the scene-pickle file
    with open(points_path, 'rb') as points_fp:
        points = pickle.load(points_fp, encoding='latin1')

    # Read the labels from the pickle files
    labels = []
    for file in files_to_visualize:
        with open(file, 'rb') as labels_fp:
            _ = pickle.load(labels_fp, encoding='latin1')
            labels.append(pickle.load(labels_fp, encoding='latin1'))

    # Animate the different labels for each list of labels
    v2 = pptk.viewer(points, labels[0])
    v2.set(point_size=0.005)
    for i in range(len(labels)):
        time.sleep(3)
        colors = list(map(lambda label: g_label_colors[label], labels[i]))
        v2.attributes(colors)


if __name__ == '__main__':
    normalize_colors()
    animate_prediction_changes(scene_points, files)

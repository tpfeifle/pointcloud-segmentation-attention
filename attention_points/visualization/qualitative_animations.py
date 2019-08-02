"""
Animates rotations of the ground-truth labels of scenes, as well as of their predicted labels.
The rendered rotated views are stored as images and can be converted to videos with ``fmpeg``:
    ffmpeg -i "scene0XXX_0X/frame_%03d.png" -c:v mpeg4 -qscale:v 0 -r 24 scene0XXX_0X.mp4

Inputs:
    - Points of a scene as .npy file (Nx3)
    - Labels (ground truth) of a scene as .npy file (Nx1)
    - Predicted labels of a scene as .npy file (Nx1)

"""

import numpy as np
import pptk
import os
from typing import List

# Scenes to be animated
scene_names = ["scene0488_00", "scene0063_00", "scene0095_01", "scene0203_00", "scene0256_02", "scene0474_05",
               "scene0256_02", "scene0500_01", "scene0660_00", "scene0702_01", "scene0648_01"]

path_to_recordings = '/Users/tim/Downloads/recordings'

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


def load_from_scene_name(scene_name: str, pre_files_dir="/Users/tim/Downloads/") -> List[np.ndarray]:
    points = np.load(pre_files_dir + scene_name + ".npy")
    colors = np.load(pre_files_dir + scene_name + "_vh_clean_2.ply.npy")
    return [points, colors]


def load_predictions(scene_name: str, model: str, files_dir="/Users/tim/Downloads/for_visualization") -> List[
    np.ndarray]:
    """
    Loads the numpy-arrays (points and labels) for the specified model from disk

    :param scene_name: Name of the scene to be loaded
    :param files_dir: Directory where the scenes reside
    :param model: Name of the model that should be loaded, element of ['baseline', 'baseline_features', 'groundtruth']
    :return:
    """

    points = np.load(files_dir + "/points/" + scene_name + ".npy")

    if model == 'baseline':
        labels = np.load(files_dir + "/labels_baseline/" + scene_name + ".npy")
    elif model == 'baseline_features':
        labels = np.load(files_dir + "/labels/" + scene_name + ".npy")
    else:
        labels = np.load(files_dir + "/groundtruth_labels/" + scene_name + ".npy")
    labels = labels.astype(np.int32)

    # map the labels to the segmentation-colors
    colors = list(map(lambda label: g_label_colors[label], labels))
    return [points, colors]


def animate_and_store(view, points: np.ndarray, scene_name: str, sub_path: str):
    """
    Rotates each scene around its center of gravity and stores the rendered frames at the path `path_to_recordings`

    :param view: Reference to the pptk-view to be rendered
    :param points: The list of points to be rendered (needed for center of gravity) (Nx3)
    :param scene_name: Name of the scene (frames stored with this name)
    :param sub_path: sub-folder of the recordings where the frames should be stored
    :return:
    """

    center_of_gravity_x = sum([pair[0] for pair in points]) / len(points)
    center_of_gravity_y = sum([pair[1] for pair in points]) / len(points)
    center_of_gravity_z = sum([pair[2] for pair in points]) / len(points)

    poses = []
    for i in range(5):
        poses.append([center_of_gravity_x, center_of_gravity_y, center_of_gravity_z, i * np.pi / 2, np.pi / 4, 10])

    path = path_to_recordings + sub_path + "/" + scene_name
    os.mkdir(path)

    view.record(path, poses, 2 * np.arange(5))


def animate_scenes(scenes: List[str]):
    """
    Renders each of the provided scene names with pptk-viewer, rotates the scene and stores each rotated frame
    as a separate file. For each scene the ground-truth as well as the predicted scene-labels are animated and stored

    :param scenes: Array with the name of the scenes to be animated
    :return:
    """
    for scene_name in scenes:
        # Read ground truth data
        points_gt, colors_gt = load_predictions(scene_name, 'groundtruth')
        v_gt = pptk.viewer(points_gt, colors_gt)
        v_gt.set(point_size=0.009, bg_color=(1, 1, 1, 1), floor_color=(1, 1, 1, 1), show_axis=False, show_grid=False,
                 r=10)

        # Read prediction data
        points, colors = load_predictions(scene_name, 'baseline_features')
        for idx, color in enumerate(colors_gt):
            if color[0] == 0:
                colors[idx] = color

        v2 = pptk.viewer(points, colors)
        v2.set(point_size=0.009, bg_color=(1, 1, 1, 1), floor_color=(1, 1, 1, 1), show_axis=False, show_grid=False,
               r=10)

        # Animates the scenes by rotating it and storing each frame in the provided path
        animate_and_store(v_gt, points_gt, scene_name, '/groundtruth')
        animate_and_store(v2, points, scene_name, '/predictions')


if __name__ == '__main__':
    normalize_colors()
    animate_scenes(scene_names)

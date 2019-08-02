import numpy as np
import pptk
import pickle
import os
import time
from typing import Generator, List

# wall, floor, desk
g_label_names = ['unannotated',
                 'wall',
                 'floor',
                 'cabinet',
                 'bed',
                 'chair',
                 'sofa',
                 'table',
                 'door',
                 'window',
                 'bookshelf',
                 'picture',
                 'counter',
                 'desk',
                 'curtain',
                 'refrigerator',
                 'shower curtain',
                 'toilet',
                 'sink',
                 'bathtub',
                 'otherfurniture']
g_label_colors = [
    [0, 0, 0],
    [173, 198, 232],
[151,223,137],
[30, 120, 180],
[255, 188, 119],
[188, 189, 36],
[140, 86, 74],
[255, 152, 151],
[213, 38, 40],
[197, 175, 213],
[148, 103, 188],
[197, 156, 148],
[24, 190, 208],
[247, 183, 210],
[218, 219, 141],
[254, 127, 11],
[158, 218, 229],
[43, 160, 45],
[111, 128, 144],
[227, 120, 193],
[82, 82, 163]
]

for idx, color in enumerate(g_label_colors):
    g_label_colors[idx] = [color[0]/255.0, color[1]/255.0, color[2]/255.0]


def load_from_scene_name(scene_name, pre_files_dir="/Users/tim/Downloads/") -> List[np.ndarray]:
    points = np.load(pre_files_dir + scene_name + ".npy")
    # labels = np.load(pre_files_dir + "labels/" + scene_name + ".npy")
    colors = np.load(pre_files_dir + scene_name + "_vh_clean_2.ply.npy")
    # normals = np.load(pre_files_dir + "normals/" + scene_name + "_vh_clean_2.ply.npy")
    # for i in [labels, colors, normals]:
    #    assert len(i) == len(points)
    return [points, colors]

def label_map_more_paraemters(labels):
    map_values = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12, 14: 13, 16: 14, 24: 15,
                  28: 16,
                  33: 17, 34: 18, 36: 19, 39: 20}
    mapped_labels = np.array(list(map(lambda label: map_values.get(label, 0), labels)))
    return mapped_labels

def load_from_pred(scene_name, pre_files_dir="/Users/tim/Downloads/for_visualization", baseline=False) -> List[np.ndarray]:
    points = np.load(pre_files_dir + "/points/" + scene_name + ".npy")
    if baseline:
        labels = np.load(pre_files_dir + "/labels_baseline/" + scene_name + ".npy")
    else:
        labels = np.load(pre_files_dir + "/labels/" + scene_name + ".npy")
    labels = labels.astype(np.int32)
    #labels = label_map_more_paraemters(labels).astype(np.int32)
    colors = list(map(lambda label: g_label_colors[label], labels))
    return [points, colors]

def load_from_groundtruth(scene_name, pre_files_dir="/Users/tim/Downloads/for_visualization") -> List[np.ndarray]:
    points = np.load(pre_files_dir + "/points/" + scene_name + ".npy")
    labels = np.load(pre_files_dir + "/groundtruth_labels/" + scene_name + ".npy")
    labels = labels.astype(np.int32)
    colors = list(map(lambda label: g_label_colors[label], labels))
    return [points, colors]

def animate_and_store(view, points, scene_name, sub_path):
    center_of_gravity_x = sum([pair[0] for pair in points])/len(points)
    center_of_gravity_y = sum([pair[1] for pair in points])/len(points)
    center_of_gravity_z = sum([pair[2] for pair in points])/len(points)
    poses = []
    poses.append([center_of_gravity_x, center_of_gravity_y, center_of_gravity_z, 0 * np.pi/2, np.pi/4, 10])
    poses.append([center_of_gravity_x, center_of_gravity_y, center_of_gravity_z, 1 * np.pi/2, np.pi/4, 10])
    poses.append([center_of_gravity_x, center_of_gravity_y, center_of_gravity_z, 2 * np.pi/2, np.pi/4, 10])
    poses.append([center_of_gravity_x, center_of_gravity_y, center_of_gravity_z, 3 * np.pi/2, np.pi/4, 10])
    poses.append([center_of_gravity_x, center_of_gravity_y, center_of_gravity_z, 4 * np.pi/2, np.pi/4, 10])
    #view.play(poses, 2 * np.arange(5), repeat=True, interp='linear')
    path = '/Users/tim/Downloads/recordings'+sub_path+"/"+scene_name
    os.mkdir(path)
    print(path)
    view.record(path, poses, 2 * np.arange(5))

# nice: 256_02, 660_00, 647_01
scene_names = [#"scene0488_00", "scene0063_00", "scene0095_01", "scene0203_00", "scene0256_02", "scene0474_05",
               "scene0256_02", "scene0500_01", "scene0660_00", "scene0702_01", "scene0648_01"]
#scene_name =
'''
drwxr-xr-x  195 tim  staff      6240 Jul 26 10:51 scene0063_00
drwxr-xr-x  195 tim  staff      6240 Jul 26 10:52 scene0095_01
drwxr-xr-x  195 tim  staff      6240 Jul 26 10:53 scene0203_00
drwxr-xr-x  195 tim  staff      6240 Jul 26 10:07 scene0256_02
drwxr-xr-x  195 tim  staff      6240 Jul 26 10:54 scene0474_05
drwxr-xr-x  195 tim  staff      6240 Jul 26 10:04 scene0488_00
'''
def do_it():
    for scene_name in scene_names:
        points_gt, colors_gt = load_from_groundtruth(scene_name)
        v_gt = pptk.viewer(points_gt, colors_gt)
        v_gt.set(point_size=0.009, bg_color=(1, 1, 1, 1), floor_color=(1, 1, 1, 1), show_axis=False, show_grid=False, r=10)

        '''points, colors = load_from_pred(scene_name)

        for idx, color in enumerate(colors_gt):
            if color[0] == 0:
                colors[idx] = color

        v2 = pptk.viewer(points, colors)
        v2.set(point_size=0.009, bg_color=(1, 1, 1, 1), floor_color=(1, 1, 1, 1), show_axis=False, show_grid=False, r=10)'''

        points, colors = load_from_pred(scene_name, baseline=True)

        for idx, color in enumerate(colors_gt):
            if color[0] == 0:
                colors[idx] = color

        v2 = pptk.viewer(points, colors)
        v2.set(point_size=0.009, bg_color=(1, 1, 1, 1), floor_color=(1, 1, 1, 1), show_axis=False, show_grid=False,
               r=10)


        animate_and_store(v_gt, points_gt, scene_name, '/groundtruth')
        animate_and_store(v2, points, scene_name, '/predictions')
do_it()
print(1)
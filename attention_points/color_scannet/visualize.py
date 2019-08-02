import os
import pickle

import numpy as np
import pptk

g_label_names = ['unannotated', 'wall', 'floor', 'chair', 'table', 'desk', 'bed', 'bookshelf', 'sofa', 'sink',
                 'bathtub', 'toilet', 'curtain', 'counter', 'door', 'window', 'shower curtain', 'refridgerator',
                 'picture', 'cabinet', 'otherfurniture']
g_label_colors = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1], [0, 0, 0.5],
                  [0, 0.5, 0], [0, 0.5, 0.5], [0.5, 0, 0], [0.5, 0, 0.5], [0.5, 0.5, 0], [0.5, 0.5, 0.5], [0, 0, 0.2],
                  [0, 0.2, 0], [0, 0.2, 0.2], [0.2, 0, 0], [0.2, 0, 0.2], [0.2, 0.2, 0]]

# Visualize ground truth
'''output = os.path.join('/tmp/batch.pickle')
with open(output, 'rb') as wfp:
    points = pickle.load(wfp, encoding='latin1')
    segments = pickle.load(wfp, encoding='latin1')
x = points[0]
colors = list(map(lambda label: g_label_colors[int(label)], segments[0]))
v = pptk.viewer(points[0], colors)
v.set(point_size=0.005)'''

# Visualize prediction
output2 = os.path.join('/tmp/to_visualize_pointnet_normal_269.pickle')
with open(output2, 'rb') as wfp2:
    points = pickle.load(wfp2, encoding='latin1')[0]
    labels = pickle.load(wfp2, encoding='latin1')[0]

print(np.max(labels))
# colors = list(map(lambda color: color/255, colors))
colors = list(map(lambda label: g_label_colors[label], labels))
v2 = pptk.viewer(points, colors)
v2.set(point_size=0.005)
v2.set(point_size=0.005, bg_color=(0, 0, 0, 0), floor_color=(0, 0, 0, 0), show_axis=False)
poses = []
poses.append([0, 0, 0, 0 * np.pi / 2, np.pi / 4, 10])
poses.append([0, 0, 0, 1 * np.pi / 2, np.pi / 4, 10])
poses.append([0, 0, 0, 2 * np.pi / 2, np.pi / 4, 10])
poses.append([0, 0, 0, 3 * np.pi / 2, np.pi / 4, 10])
poses.append([0, 0, 0, 4 * np.pi / 2, np.pi / 4, 10])
v2.play(poses, 2 * np.arange(5), repeat=True, interp='linear')
print(1)

import numpy as np
import pptk
import pickle
import os
import time
g_label_names = ['unannotated', 'wall', 'floor', 'chair', 'table', 'desk', 'bed', 'bookshelf', 'sofa', 'sink', 'bathtub', 'toilet', 'curtain', 'counter', 'door', 'window', 'shower curtain', 'refridgerator', 'picture', 'cabinet', 'otherfurniture',
                 'unannotated', 'wall', 'floor', 'chair', 'table', 'desk', 'bed', 'bookshelf', 'sofa', 'sink',
                 'bathtub', 'toilet', 'curtain', 'counter', 'door', 'window', 'shower curtain', 'refridgerator',
                 'picture', 'cabinet', 'otherfurniture']
g_label_colors = [[1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0.2, 0.5, 0],
                  [1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
                  [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
                  [0, 0, 0], [0, 0, 0], [0.2, 0.5, 0]]

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
output2 = os.path.join('/tmp/to_visualize1562169522_74.pickle')
with open(output2, 'rb') as wfp2:
    points = pickle.load(wfp2, encoding='latin1')
    labels = pickle.load(wfp2, encoding='latin1')
    colors = pickle.load(wfp2, encoding='latin1')


files = [os.path.join('/tmp/to_visualize1562169522_19.pickle'),
         os.path.join('/tmp/to_visualize1562169522_29.pickle'),
         os.path.join('/tmp/to_visualize1562169522_74.pickle')]
other_labels = []
with open(files[0], 'rb') as wfp2:
    _ = pickle.load(wfp2, encoding='latin1')
    other_labels.append(pickle.load(wfp2, encoding='latin1'))
with open(files[1], 'rb') as wfp2:
    _ = pickle.load(wfp2, encoding='latin1')
    other_labels.append(pickle.load(wfp2, encoding='latin1'))
with open(files[2], 'rb') as wfp2:
    _ = pickle.load(wfp2, encoding='latin1')
    other_labels.append(pickle.load(wfp2, encoding='latin1'))
print(np.max(labels))
# colors = list(map(lambda color: color/255, colors))
colors = list(map(lambda label: g_label_colors[label], labels))
v2 = pptk.viewer(points, colors)
v2.set(point_size=0.005)
for i in range(len(other_labels)):
    time.sleep(3)
    colors = list(map(lambda label: g_label_colors[label], other_labels[i]))
    v2.attributes(colors)

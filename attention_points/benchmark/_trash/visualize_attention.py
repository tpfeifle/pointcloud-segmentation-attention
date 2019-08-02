import os
import pickle

import numpy as np
import pptk

# Visualize prediction
output2 = os.path.join('/tmp/attention_weights.pickle')
with open(output2, 'rb') as wfp2:
    p = pickle.load(wfp2, encoding='latin1')[0, :, 1, 0]  # use only mha with index 1
    xyz = pickle.load(wfp2, encoding='latin1')[0]
    idx = pickle.load(wfp2, encoding='latin1')[0]

# weight = np.sum(p, axis=1)
weight = p
print(weight)
print(np.min(weight))
print(np.max(weight))

'''Result := ((Input - InputLow) / (InputHigh - InputLow))
          * (OutputHigh - OutputLow) + OutputLow;'''


def toColorRange(p):
    val = (p - np.min(weight)) / (np.max(weight) - np.min(weight)) * 255
    return val


def idxToCoordinates(id):
    return xyz[id]


colors = list(map(lambda i: toColorRange(i), weight))
coordinates = np.array(list(map(lambda id: idxToCoordinates(id), idx[:, ])))
all_coordinates = []
for cor in coordinates:
    for a in cor:
        all_coordinates.append(a)
all_colors = []
for col in colors:
    for a in col:
        all_colors.append(a)
v2 = pptk.viewer(all_coordinates, all_colors)
v2.color_map('gray')
v2.set(point_size=0.005, bg_color=(0, 0, 0, 0), floor_color=(0, 0, 0, 0), show_axis=False)

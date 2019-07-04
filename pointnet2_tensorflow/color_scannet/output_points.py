import pptk
import numpy as np
import pickle
import time

g_label_names = ['unannotated', 'wall', 'floor', 'chair', 'table', 'desk', 'bed', 'bookshelf', 'sofa', 'sink',
                 'bathtub', 'toilet', 'curtain', 'counter', 'door', 'window', 'shower curtain', 'refridgerator',
                 'picture', 'cabinet', 'otherfurniture']
g_label_colors = [[1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
                  [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
                  [0, 0, 0], [0, 0, 0], [0.2, 0.5, 0]]
outfile = '/tmp/pycharm_project_250/to_visualize%s' % int(time.time())


def output(points, labels, colors=[], index=0):
    with open(outfile + '_' + str(index) + '.pickle', 'wb') as fp:
        pickle.dump(points, fp)
        pickle.dump(labels, fp)
        pickle.dump(colors, fp)

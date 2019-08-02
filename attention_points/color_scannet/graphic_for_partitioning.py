from itertools import product, combinations

import matplotlib.pyplot as plt
import numpy as np

points = np.load('/Users/tim/Downloads/scene0549_01.npy')


def pyplot_draw_point_cloud(points, output_filename):
    """ points is a Nx3 numpy array """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=0.001)
    ax.view_init(30, 130)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim3d(1, 7)
    ax.set_ylim3d(6, 8)
    ax.set_zlim3d(0, 2)

    # draw cube
    r = [2, 3]
    for s, e in combinations(np.array(list(product(r, r, r))), 2):
        if np.sum(np.abs(s - e)) == r[1] - r[0]:
            ax.plot3D(*zip(s, e), color="b")
    r = [3, 3]
    for s, e in combinations(np.array(list(product(r, r, r))), 2):
        if np.sum(np.abs(s - e)) == r[1] - r[0]:
            ax.plot3D(*zip(s, e), color="g")

    plt.show()


pyplot_draw_point_cloud(points, '')

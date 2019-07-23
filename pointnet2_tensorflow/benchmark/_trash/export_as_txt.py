# Example script to output evaluation format given a volumetric grid of predictions.
# Input:
#   - prediction grid as path to .npy np array
#   - world2grid transform as path to .npy 4x4 np array
#   - path to the corresponding *_vh_clean_2.ply mesh
#   - output file to write evaluation format .txt file
#
# example usage: export_semantic_label_grid_for_evaluation.py --grid_file [path to predicted grid] --world2grid_file [path to world to grid] --output_file [output file] --mesh_file [path to corresponding mesh file]

# python imports
import math
import os, sys, argparse
import inspect
import numpy as np

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from benchmark import util_3d


def main():
    predictions = np.load("/Users/tim/Downloads/pred_scene0568_01.npy")
    output_file = "predictions0568_01.txt"
    util_3d.export_ids(output_file, predictions)


if __name__ == '__main__':
    main()
# Example of the output format for evaluation for 3d semantic label and instance prediction.
# Exports a train scan in the evaluation format using:
#   - the *_vh_clean_2.ply mesh
#   - the labels defined by the *.aggregation.json and *_vh_clean_2.0.010000.segs.json files
#
# example usage: export_train_mesh_for_evaluation.py --scan_path [path to scan data] --output_file [output file] --type label
# Note: technically does not need to load in the ply file, since the ScanNet annotations are defined against the mesh vertices, but we load it in here as an example.

import json
# python imports
import os

import numpy as np

from attention_points.visualization import util, util_3d

TASK_TYPES = {'label', 'instance'}

'''parser = argparse.ArgumentParser()
parser.add_argument('--scan_path', required=True, help='path to scannet scene (e.g., data/ScanNet/v2/scene0000_00')
parser.add_argument('--output_file', required=True, help='output file')
parser.add_argument('--label_map_file', required=True, help='path to scannetv2-labels.combined.tsv')
parser.add_argument('--type', required=True, help='task type [label or instance]')
opt = parser.parse_args()
assert opt.type in TASK_TYPES'''


def read_aggregation(filename):
    assert os.path.isfile(filename)
    object_id_to_segs = {}
    label_to_segs = {}
    with open(filename) as f:
        data = json.load(f)
        num_objects = len(data['segGroups'])
        for i in range(num_objects):
            object_id = data['segGroups'][i]['objectId'] + 1  # instance ids should be 1-indexed
            label = data['segGroups'][i]['label']
            segs = data['segGroups'][i]['segments']
            object_id_to_segs[object_id] = segs
            if label in label_to_segs:
                label_to_segs[label].extend(segs)
            else:
                label_to_segs[label] = segs
    return object_id_to_segs, label_to_segs


def read_segmentation(filename):
    assert os.path.isfile(filename)
    seg_to_verts = {}
    with open(filename) as f:
        data = json.load(f)
        num_verts = len(data['segIndices'])
        for i in range(num_verts):
            seg_id = data['segIndices'][i]
            if seg_id in seg_to_verts:
                seg_to_verts[seg_id].append(i)
            else:
                seg_to_verts[seg_id] = [i]
    return seg_to_verts, num_verts


def export(agg_file, seg_file, label_map_file, type, output_file):
    label_map = util.read_label_mapping(label_map_file, label_from='raw_category', label_to='nyu40id')
    object_id_to_segs, label_to_segs = read_aggregation(agg_file)
    seg_to_verts, num_verts = read_segmentation(seg_file)
    label_ids = np.zeros(shape=(num_verts), dtype=np.uint32)  # 0: unannotated
    for label, segs in label_to_segs.items():
        label_id = label_map[label]
        for seg in segs:
            verts = seg_to_verts[seg]
            label_ids[verts] = label_id
    if type == 'label':
        util_3d.export_ids(output_file, label_ids)
    else:
        print("ERROR was raised")


def main():
    scan_path = "/Users/tim/Downloads/scannet_downloads/scans/"  # scene0568_01/
    output_path = "/Users/tim/Downloads/scannet_groundtruth/"
    scene_folders = [dI for dI in os.listdir(scan_path) if os.path.isdir(os.path.join(scan_path, dI))]
    for folder in scene_folders:
        output_file = output_path + folder + ".txt"  # ""truth0568_01.txt"
        label_map_file = "/Users/tim/Downloads/scannet_downloads/scannetv2-labels.combined.tsv"
        type = "label"
        scan_name = folder  # os.path.split(scan_path)[-1]
        scene_path = os.path.join(scan_path, folder)
        agg_file = os.path.join(scene_path, scan_name + '.aggregation.json')
        seg_file = os.path.join(scene_path, scan_name + '_vh_clean_2.0.010000.segs.json')
        export(agg_file, seg_file, label_map_file, type, output_file)


if __name__ == '__main__':
    main()

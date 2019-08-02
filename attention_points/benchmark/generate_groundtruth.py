"""
Exports the groundtruth labels for the scans in the evaluation format.
Adapted from ScanNet benchmark scripts: https://github.com/ScanNet/ScanNet/tree/master/BenchmarkScripts
    Authors of ScanNet:
    Dai, Angela and Chang, Angel X. and Savva, Manolis and Halber, Maciej and Funkhouser, Thomas and Niessner, Matthias

Input:

- the *_vh_clean_2.ply mesh
- the labels defined by the *.aggregation.json and *_vh_clean_2.0.010000.segs.json files

"""

import json
import os
import numpy as np
import csv
from typing import Dict

scan_path = "/Users/tim/Downloads/scannet_downloads/scans/"
output_path = "/Users/tim/Downloads/scannet_groundtruth/"
label_map_file = "/Users/tim/Downloads/scannet_downloads/scannetv2-labels.combined.tsv"


def read_aggregation(filename: str):
    """
    Read the aggregation data for the scene

    :param filename: Path to the aggregation data
    :return:
    """
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


def read_segmentation(filename: str):
    """
    Read the segmentation data for the scene

    :param filename: Path to the segmentation data
    :return:
    """
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


def represents_int(s) -> bool:
    """
    Makes sure that string s represents an int
    :param s: String to be checked
    :return: Whether or not string represents an int
    """
    try:
        int(s)
        return True
    except ValueError:
        return False


def read_label_mapping(filename: str, label_from='raw_category', label_to='nyu40id') -> Dict:
    """
    Read the label mapping from the provided filename

    :param filename: Path to the label mapping
    :param label_from: Name of the field that contains the label
    :param label_to: Name of the value to map to
    :return: Remapped labels
    """
    assert os.path.isfile(filename)
    mapping = dict()
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        for row in reader:
            mapping[row[label_from]] = int(row[label_to])
    # if ints convert
    if represents_int(list(mapping.keys())[0]):
        mapping = {int(k): v for k, v in mapping.items()}
    return mapping


def export_ids(filename: str, ids: np.ndarray):
    """
    Export the provided ids to the provided filename
    :param filename: Path to export to
    :param ids: Ids that should be exported
    :return:
    """
    with open(filename, 'w') as f:
        for id in ids:
            f.write('%d\n' % id)


def export(agg_file: str, seg_file: str, label_map: str, output_file: str):
    """
    Exports the specified groundtruth scene in the benchmark evaluation format

    :param agg_file: Path to the aggregation file of the scene
    :param seg_file: Path to the segmentation file of the scene
    :param label_map: Path to the label_map (nyu40-scannet)
    :param output_file: Path to which the output should be written
    :return:
    """
    label_map = read_label_mapping(label_map, label_from='raw_category', label_to='nyu40id')
    object_id_to_segs, label_to_segs = read_aggregation(agg_file)
    seg_to_verts, num_verts = read_segmentation(seg_file)
    label_ids = np.zeros(shape=num_verts, dtype=np.uint32)  # 0: unannotated
    for label, segs in label_to_segs.items():
        label_id = label_map[label]
        for seg in segs:
            verts = seg_to_verts[seg]
            label_ids[verts] = label_id
    export_ids(output_file, label_ids)


def main():
    """
    Exports for each groundtruth-label file in the provided folder the labels in the format required for
    the ScanNet benchmark
    :return:
    """
    scene_folders = [dI for dI in os.listdir(scan_path) if os.path.isdir(os.path.join(scan_path, dI))]
    for scan_name in scene_folders:
        output_file = output_path + scan_name + ".txt"
        scene_path = os.path.join(scan_path, scan_name)
        agg_file = os.path.join(scene_path, scan_name + '.aggregation.json')
        seg_file = os.path.join(scene_path, scan_name + '_vh_clean_2.0.010000.segs.json')
        export(agg_file, seg_file, label_map_file, output_file)


if __name__ == '__main__':
    main()

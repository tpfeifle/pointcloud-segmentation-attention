"""
Evaluates semantic label task

Adapted from ScanNet benchmark scripts: https://github.com/ScanNet/ScanNet/tree/master/BenchmarkScripts
    Authors of ScanNet:
    Dai, Angela and Chang, Angel X. and Savva, Manolis and Halber, Maciej and Funkhouser, Thomas and Niessner, Matthias

Input:

  - path to .txt prediction files
  - path to .txt ground truth files
  - output file to write results to

Note that only the valid classes are used for evaluation,
i.e., any ground truth label not in the valid label set
is ignored in the evaluation.
"""

import inspect
import os
import sys
import numpy as np
from typing import Dict, List

pred_path = "/home/tim/results/predictions_colors"
gt_path = "/home/tim/results/groundtruth"
output_file = "/home/tim/results/results_colors.txt"

try:
    from itertools import izip
except ImportError:
    izip = zip

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)


def load_ids(filename: str):
    """
    Read the predicted label ids from the specified filename

    :param filename: Name of the label file
    :return:
    """
    ids = open(filename).read().splitlines()
    ids = np.array(ids, dtype=np.int64)
    return ids


CLASS_LABELS = ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture',
                'counter', 'desk', 'curtain', 'refrigerator', 'shower curtain', 'toilet', 'sink', 'bathtub',
                'otherfurniture']
VALID_CLASS_IDS = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])
UNKNOWN_ID = np.max(VALID_CLASS_IDS) + 1


def evaluate_scan(pred_file: str, gt_file: str, confusion: np.ndarray):
    """
    Update confusion matrix for the specified scene

    :param pred_file: file containing the label predictions for one scene
    :param gt_file: file containing the label groundtruth for one scene
    :param confusion: confusion matrix to be updated by this scene
    :return:
    """
    try:
        pred_ids = load_ids(pred_file)
    except Exception as e:
        print('unable to load ' + pred_file + ': ' + str(e))
    try:
        gt_ids = load_ids(gt_file)
    except Exception as e:
        print('unable to load ' + gt_file + ': ' + str(e))
    # sanity checks
    if not pred_ids.shape == gt_ids.shape:
        print('%s: number of predicted values does not match number of vertices' % pred_file)
    for (gt_val, pred_val) in izip(gt_ids.flatten(), pred_ids.flatten()):
        if gt_val not in VALID_CLASS_IDS:
            continue
        if pred_val not in VALID_CLASS_IDS:
            pred_val = UNKNOWN_ID
        confusion[gt_val][pred_val] += 1


def get_iou(label_id: int, confusion: np.ndarray):
    """
    Determine the IoU for the specified label_id by consulting the confusion matrix

    :param label_id: Label for which the IoU should be determined
    :param confusion: Confusion matrix (from the evaluation)
    :return:
    """
    if not label_id in VALID_CLASS_IDS:
        return float('nan')

    # #true positives
    tp = np.longlong(confusion[label_id, label_id])
    # #false negatives
    fn = np.longlong(confusion[label_id, :].sum()) - tp
    # #false positives
    not_ignored = [l for l in VALID_CLASS_IDS if not l == label_id]
    fp = np.longlong(confusion[not_ignored, label_id].sum())

    denom = (tp + fp + fn)
    if denom == 0:
        return float('nan')
    return (float(tp) / denom, tp, denom)


def write_result_file(confusion: np.ndarray, ious: Dict[int, float], filename: str):
    """
    Write the evaluation result (IoUs and confusion matrix) to the specified filename

    :param confusion: Confusion matrix (from the evaluation)
    :param ious: IoUs for each label
    :param filename: Filename to output the evaluation results to
    :return:
    """
    with open(filename, 'w') as f:
        f.write('iou scores\n')
        for i in range(len(VALID_CLASS_IDS)):
            label_id = VALID_CLASS_IDS[i]
            label_name = CLASS_LABELS[i]
            iou = ious[label_name][0]
            f.write('{0:<14s}({1:<2d}): {2:>5.3f}\n'.format(label_name, label_id, iou))
        f.write('\nconfusion matrix\n')
        f.write('\t\t\t')
        for i in range(len(VALID_CLASS_IDS)):
            # f.write('\t{0:<14s}({1:<2d})'.format(CLASS_LABELS[i], VALID_CLASS_IDS[i]))
            f.write('{0:<8d}'.format(VALID_CLASS_IDS[i]))
        f.write('\n')
        for r in range(len(VALID_CLASS_IDS)):
            f.write('{0:<14s}({1:<2d})'.format(CLASS_LABELS[r], VALID_CLASS_IDS[r]))
            for c in range(len(VALID_CLASS_IDS)):
                f.write('\t{0:>5.3f}'.format(confusion[VALID_CLASS_IDS[r], VALID_CLASS_IDS[c]]))
            f.write('\n')
    print('wrote results to', filename)


def evaluate(pred_files: List[str], gt_files: List[str], output_file_path: str):
    """
    Calculate the IoU for each label in each of the pred_files and write the summary output to the output_file_path

    :param pred_files: Files containing the predictions for each scene
    :param gt_files: Files containing the groundtruth for each scene
    :param output_file_path: File to which the output should be written
    :return:
    """
    max_id = UNKNOWN_ID
    confusion = np.zeros((max_id + 1, max_id + 1), dtype=np.ulonglong)

    print('evaluating', len(pred_files), 'scans...')
    for i in range(len(pred_files)):
        evaluate_scan(pred_files[i], gt_files[i], confusion)
        sys.stdout.write("\rscans processed: {}".format(i + 1))
        sys.stdout.flush()
    print('')

    class_ious = {}
    for i in range(len(VALID_CLASS_IDS)):
        label_name = CLASS_LABELS[i]
        label_id = VALID_CLASS_IDS[i]
        class_ious[label_name] = get_iou(label_id, confusion)
    print('classes          IoU')
    print('----------------------------')
    for i in range(len(VALID_CLASS_IDS)):
        label_name = CLASS_LABELS[i]
        print('{0:<14s}: {1:>5.3f}   ({2:>6d}/{3:<6d})'.format(label_name, class_ious[label_name][0],
                                                               class_ious[label_name][1], class_ious[label_name][2]))
    write_result_file(confusion, class_ious, output_file_path)


def main():
    """
    Evaluate the IoU scores of the predicted labels for all the scenes by comparing with the groundtruth labels
    :return:
    """
    pred_files = [f for f in os.listdir(pred_path) if f.endswith('.txt')]
    gt_files = []
    if len(pred_files) == 0:
        print('No result files found.')
    for i in range(len(pred_files)):
        gt_file = os.path.join(gt_path, pred_files[i])
        if not os.path.isfile(gt_file):
            print('Result file {} does not match any gt file'.format(pred_files[i]))
        gt_files.append(gt_file)
        pred_files[i] = os.path.join(pred_path, pred_files[i])

    # evaluate
    evaluate(pred_files, gt_files, output_file)


if __name__ == '__main__':
    main()

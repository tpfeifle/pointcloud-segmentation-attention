"""
Predict the labels for all points in all scenes of the validation dataset.
Here we apply our bigger cuboids to get better predictions of the points at the border of subsets.
"""

import numpy as np
import tensorflow as tf
from attention_points.scannet_dataset import precompute_dataset
from typing import Tuple, List

N_POINTS = 8192

output_path_predictions = "/home/tim/results/for_visualization"
baseline_path = "/home/tim/training_log/pointnet_and_features/long_run1563786310_continued_train"
color_path = "/home/tim/training_log/baseline/color_test_run_1563999967_train"
advanced_baseline_path = "/home/tim/training_log/baseline/long_run1563533884_train"


def map_back(values: np.ndarray, original_idx: np.ndarray, mask: np.ndarray, res_shape) -> np.ndarray:
    """
    Applies the subset masks to ignore the predictions for points outside of the cuboid (dxdxd') that where only
    added to provide context to the points at the border of the cuboid. Because we choose random points by shuffling
    the points we here remap the predictions to their original ids (inverse-shuffle).

    :param values: An array of values (e.g. labels, ...)
    :param original_idx: The ids the points had before being shuffled. Used to inverse-shuffle the points here
    :param mask: The mask to remove the points of the larger cuboid that were only added to support the prediction of
                 the border points of the original cuboid (smarter scene subsets)
    :param res_shape: Shape into which we are remapping
    :return: The remapped values in the shape res_shape
    """
    values = values[mask]
    original_idx = original_idx[mask]

    res = np.zeros(res_shape)
    res[original_idx] = values
    return res


def map_to_nyu40(labels: np.ndarray) -> np.ndarray:
    """
    Mapping the ScanNet labels (1-20) back to the original NYU-40 labels

    :param labels: labels in the ScanNet format (1-20) in the shape (Nx1)
    :return: labels in the NYU-40 format (Nx1)
    """
    labels = np.array(labels)
    nyu40_labels = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12, 13: 14, 14: 16,
                    15: 24, 16: 28,
                    17: 33, 18: 34, 19: 36, 20: 39}
    for i in range(len(labels)):
        labels[i] = nyu40_labels.get(labels[i], 1)
    return labels


def export_ids(filename: str, ids: List):
    """
    Export the provided ids to the provided filename
    :param filename: Path to export to
    :param ids: Ids that should be exported
    :return:
    """
    with open(filename, 'w') as f:
        for id in ids:
            f.write('%d\n' % id)


def get_validation_data(sess, test=False) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Create the validation dataset from the evaluation generator. This generator generates subsets containing
    each point of each validation scene and uses smart-scene subsets for better predictions

    :param sess: tensorflow session
    :param test: Set to true to load the test scenes instead of the validation scenes
    :return: Labels (Nx1), Coordinates (Nx3), Features(Nxk), Scene Name and the original point ids (Nx1) and mask (Nx1)
             for remapping
    """
    if test:
        val_data = precompute_dataset.test_dataset_from_generator()
    else:
        val_data = precompute_dataset.eval_dataset_from_generator()
    val_data = val_data.batch(1)
    val_iterator = tf.data.Iterator.from_structure(val_data.output_types, val_data.output_shapes)
    val_data_init = val_iterator.make_initializer(val_data)
    sess.run(val_data_init)
    points, labels, colors, normals, scene_name, mask, points_orig_idxs = val_iterator.get_next()
    colors = tf.div(tf.cast(colors, tf.float32), tf.constant(255, dtype=tf.float32))
    val_features = tf.concat([tf.cast(colors, tf.float32), normals], 2)
    val_coordinates = points
    val_labels = labels
    return val_labels, val_coordinates, val_features, scene_name, points_orig_idxs, mask


def generate_predictions(model_save_path, features=True):
    """
    Generate the predictions for each point in each of the validation scenes and outputs the results in the
    required ScanNet benchmark format

    :param model_save_path: Path to the saved model that should be restored for the predictions
    :param features: Whether or not the models uses additional features (colors, normals) as input or not
    :return:
    """
    tf.Graph().as_default()
    tf.device('/gpu:0')
    tf.reset_default_graph()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    val_labels, val_coordinates, val_features, scene_name, points_orig_idxs, mask = get_validation_data(sess)

    # define model and metrics
    is_training_pl = tf.Variable(False)

    if features:
        import attention_points.models.pointnet2_sem_seg_features as model
        val_pred, _ = model.get_model(val_coordinates, val_features, is_training_pl, 21)
    else:
        import models.pointnet2_sem_seg as model
        val_pred, _ = model.get_model(val_coordinates, is_training_pl, 21)

    # initialize variables
    variable_init = tf.global_variables_initializer()

    sess.run(variable_init)
    sess.run(tf.local_variables_initializer())

    # Load trained model
    saver = tf.train.Saver()
    get_checkpoint = tf.train.latest_checkpoint(model_save_path)
    saver.restore(sess, get_checkpoint)

    print("starting evaluation for all batches")
    all_pred = []
    all_labels = []
    all_points = []
    all_masks = []
    all_points_orig_idxs = []
    current_scene = ""
    while True:
        pr_res, labels_res, points_res, scene, points_orig, masks = sess.run(
            [val_pred, val_labels, val_coordinates, scene_name, points_orig_idxs, mask])

        # remove single batch dimension
        predictions_res = np.squeeze(pr_res)
        max_pred = np.argmax(predictions_res, axis=1)
        points_res = np.squeeze(points_res)
        labels_res = np.squeeze(labels_res)
        points_orig = np.squeeze(points_orig)
        masks = np.squeeze(masks)

        if scene != current_scene and current_scene != "":
            current_scene_name = current_scene[0].decode('ascii')
            print(
                "Predicted all points for scene %s and will now evaluate this scene`s predictions" % current_scene_name)
            all_points = np.concatenate(all_points)
            all_pred = np.concatenate(all_pred)
            all_labels = np.concatenate(all_labels)
            all_masks = np.concatenate(all_masks)
            all_masks = np.array(all_masks, dtype=bool)
            all_points_orig_idxs = np.concatenate(all_points_orig_idxs)

            restored_labels = map_back(all_labels, all_points_orig_idxs, all_masks,
                                       (len(np.unique(all_points_orig_idxs))))
            restored_points = map_back(all_points, all_points_orig_idxs, all_masks,
                                       (len(np.unique(all_points_orig_idxs)), 3))
            remapped_pred = map_back(all_pred, all_points_orig_idxs, all_masks,
                                     (len(np.unique(all_points_orig_idxs))))

            # Storing the predictions
            np.save(output_path_predictions + "/points/%s.npy" % current_scene_name, restored_points)
            np.save(output_path_predictions + "/labels/%s.npy" % current_scene_name, remapped_pred)
            np.save(output_path_predictions + "/groundtruth_labels/%s.npy" % current_scene_name, restored_labels)

            remapped_pred = map_to_nyu40(remapped_pred)

            # Output predictions in the ScanNet benchmark format
            output_predictions = "/home/tim/results/predictions_colors/%s.txt" % current_scene[0].decode('ascii')
            util_3d.export_ids(output_predictions, remapped_pred)

            all_pred, all_labels, all_points, all_points_orig_idxs, all_masks = [], [], [], [], []
        all_pred.append(max_pred)
        all_labels.append(labels_res)
        all_points.append(points_res)
        all_points_orig_idxs.append(points_orig)
        all_masks.append(masks)
        current_scene = scene


if __name__ == '__main__':
    generate_predictions(baseline_path)

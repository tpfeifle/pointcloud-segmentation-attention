"""
This module contains methods to transform the dataset These transformations include:
 - mapping the labels from nyu40 to range [0, 20]
 - rotating a point cloud
 - getting a random chunk of a scene
 - getting chunks for every area of a scene
 - getting all points of a scene grouped by chunks
Some of these methods are implemented in both tensorflow and numpy.
When executing on CPU the numpy versions are considerably faster.
"""
import math
from typing import Dict, Tuple

import numpy as np
import tensorflow as tf

import attention_points.scannet_dataset.generator_dataset as gd

LABEL_MAP: Dict[int, int] = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12, 14: 13,
                             16: 14, 24: 15, 28: 16, 33: 17, 34: 18, 36: 19, 39: 20}


def label_map(points: tf.Tensor, labels: tf.Tensor, colors: tf.Tensor, normals: tf.Tensor) \
        -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    tensorflow function to map labels from nyu40 to range [0, 20]

    :param points: (Nx3)
    :param labels: (N)
    :param colors: (Nx3)
    :param normals: (Nx3)
    :return: points(Nx3), mapped labels(N), colors(Nx3), normals(Nx3)
    """
    map = tf.convert_to_tensor([LABEL_MAP.get(i, 0) for i in range(41)])
    labels = tf.minimum(labels, 40)
    mapped_labels = tf.gather(map, labels)
    return points, mapped_labels, colors, normals


def label_map_numpy(points: np.ndarray, labels: np.ndarray, colors: np.ndarray, normals: np.ndarray) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    numpy function to map labels from nyu40 to range [0, 20]

    :param points: (Nx3)
    :param labels: (N)
    :param colors: (Nx3)
    :param normals: (Nx3)
    :return: points(Nx3), mapped labels(N), colors(Nx3), normals(Nx3)
    """
    map = np.array([LABEL_MAP.get(i, 0) for i in range(41)])
    labels = np.minimum(labels, 40)
    mapped_labels = map[labels]
    return points, mapped_labels, colors, normals


def label_map_more_parameters(labels: np.ndarray) -> np.ndarray:
    """
    numpy function to map labels from nyu40 to range [0, 20]

    :param labels: (N)
    :return: mapped labels (N)
    """
    mapped_labels = np.array(list(map(lambda label: LABEL_MAP.get(label, 0), labels)))
    return mapped_labels


def get_subset(points: tf.Tensor, labels: tf.Tensor, colors: tf.Tensor, normals: tf.Tensor, npoints: int = 8192) \
        -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    tensorflow function to get a random chunk of a scene which has exactly K points

    :param points: (Nx3)
    :param labels: (N)
    :param colors: (Nx3)
    :param normals: (Nx3)
    :param npoints: (K) count of points to output (default 8192)
    :return: points(Kx3), labels(K), colors(Kx3), normals(Kx3), sample_weights(K)
    """
    label_weights = [0, 2.743064592944318, 3.0830506790927132, 4.785754459526457, 4.9963745147506184,
                     4.372710774561782, 5.039124880965811, 4.86451825464344, 4.717751595568025, 4.809412839311939,
                     5.052097251455304, 5.389129668645318, 5.390614085649042, 5.127458225110977, 5.086056870814752,
                     5.3831185190895265, 5.422684124268539, 5.422955391988761, 5.433705358072363, 5.417426773812747,
                     4.870172044153657]

    coordmax = tf.reduce_max(points, axis=0)
    coordmin = tf.reduce_min(points, axis=0)

    cur_points, cur_labels, cur_colors, cur_normals = tf.zeros((1, 3)), tf.zeros((1,), dtype=tf.int32), \
                                                      tf.zeros((1, 3), dtype=tf.int32), tf.zeros((1, 3))
    mask = tf.cast(tf.zeros((1,)), tf.bool)

    def try_to_get_subset():
        # get a random area
        len = tf.reduce_sum(tf.ones_like(labels, dtype=tf.float32))
        rand_index = tf.cast(tf.random_uniform((1,), 0, len), dtype=tf.int32)
        current_center = tf.gather(points, rand_index, axis=0, name="1g")
        current_center = tf.reshape(current_center, [-1])
        current_min = current_center - tf.convert_to_tensor([0.75, 0.75, 1.5])
        current_max = current_center + tf.convert_to_tensor([0.75, 0.75, 1.5])
        current_min = tf.concat([tf.gather(current_min, [0, 1], name="2g"), tf.gather(coordmin, [2])], axis=0,
                                name="3g")
        current_max = tf.concat([tf.gather(current_max, [0, 1], name="4g"), tf.gather(coordmax, [2])], axis=0,
                                name="5g")
        # find points in area
        in_area = tf.logical_and(tf.greater_equal(points, (current_min - 0.2)),
                                 tf.greater((current_max + 0.2), points))
        where = tf.where(tf.reduce_all(in_area, axis=1))
        current_choice = tf.reshape(tf.where(tf.reduce_all(in_area, axis=1)), [-1])
        # find current points
        cur_points = tf.gather(points, current_choice, name="6g")
        cur_labels = tf.gather(labels, current_choice, name="7g")
        cur_colors = tf.gather(colors, current_choice, name="8g")
        cur_normals = tf.gather(normals, current_choice, name="9g")
        cur_len = tf.reduce_sum(tf.ones_like(cur_points))
        in_area = tf.logical_and(tf.greater_equal(cur_points, (current_min - 0.01)),
                                 tf.greater((current_max + 0.01), cur_points))
        mask = tf.reduce_all(in_area, axis=1)
        mask_indices = tf.where(mask)
        # check if subset is valid
        vidx = tf.ceil((tf.gather(cur_points, mask_indices, name="10g") - current_min) / (current_max - current_min)
                       * tf.convert_to_tensor([31.0, 31.0, 62.0]))
        vidx_0 = tf.reshape(tf.gather(tf.transpose(vidx), 0, name="11g"), [-1])
        vidx_1 = tf.reshape(tf.gather(tf.transpose(vidx), 1, name="12g"), [-1])
        vidx_2 = tf.reshape(tf.gather(tf.transpose(vidx), 2, name="13g"), [-1])
        vidx, _ = tf.unique(vidx_0 * 31.0 * 62.0 + vidx_1 * 62 + vidx_2)
        vidx_len = tf.reduce_sum(tf.ones_like(vidx))
        labeled_points = tf.cast(tf.reduce_sum(tf.cast(tf.greater(cur_labels, 0), tf.int32)), tf.float32)
        isvalid = tf.logical_and(tf.greater_equal(tf.divide(labeled_points, cur_len), 0.7),
                                 tf.greater_equal(vidx_len / 31.0 / 31.0 / 62.0, 0.02))
        return isvalid, cur_points, cur_labels, cur_colors, cur_normals, mask

    def keep_fn():
        return isvalid, cur_points, cur_labels, cur_colors, cur_normals, mask

    # try 10 times to get a valid subset, if fails, get the last
    isvalid = tf.convert_to_tensor(False)
    for i in range(10):
        isvalid, cur_points, cur_labels, cur_colors, cur_normals, mask = tf.cond(isvalid, keep_fn, try_to_get_subset)

    # get subset of correct length
    cur_len = tf.reduce_sum(tf.ones_like(cur_labels, dtype=tf.float32))
    choice = tf.cast(tf.random_uniform((npoints,), 0, cur_len), tf.int32)
    # choice = np.random.choice(len(cur_labels), npoints, replace=True)
    points = tf.gather(cur_points, choice, name="14g")
    labels = tf.gather(cur_labels, choice, name="15g")
    normals = tf.gather(cur_normals, choice, name="16g")
    colors = tf.gather(cur_colors, choice, name="17g")
    mask = tf.gather(mask, choice, name="18g")
    sample_weight = tf.gather(label_weights, labels, name="19g")
    sample_weight *= tf.cast(mask, tf.float32)
    return points, labels, colors, normals, sample_weight


def get_all_subsets_for_scene(points: tf.Tensor, labels: tf.Tensor, colors: tf.Tensor, normals: tf.Tensor,
                              npoints: int = 8192) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    tensorflow function to get chunks with K points for every area of a scene
    warning: this method is not tested yet -> use with care

    :param points: (Nx3)
    :param labels: (N)
    :param colors: (Nx3)
    :param normals: (Nx3)
    :param npoints: (K) count of points to output (default 8192)

    :return: points_stack (X,K,3), labels_stack (X,K), colors_stack (X,K,3), normals_stack (X,K,3),
        sample_weight_stack(X,K)
    """
    label_weights = tf.ones(21)  # For validation the weights keep all equal

    points_stack = tf.zeros([0, npoints, 3], dtype=tf.float32)
    labels_stack = tf.zeros([0, npoints], dtype=tf.int32)
    colors_stack = tf.zeros([0, npoints, 3], dtype=tf.int32)
    normals_stack = tf.zeros([0, npoints, 3], dtype=tf.float32)

    coordmax = tf.reduce_max(points, axis=0)
    coordmin = tf.reduce_min(points, axis=0)

    n_subvolume_x = tf.ceil((tf.gather(coordmax, [0]) - tf.gather(coordmin, [0])) / 1.5)
    n_subvolume_y = tf.ceil((tf.gather(coordmax, [1]) - tf.gather(coordmin, [1])) / 1.5)
    n_subvolume_x = tf.cast(n_subvolume_x, tf.int32)
    n_subvolume_y = tf.cast(n_subvolume_y, tf.int32)
    total_subvolumes = tf.reshape(n_subvolume_x * n_subvolume_y, [])
    current_subvolumes = tf.convert_to_tensor(0, dtype=tf.int32)

    def not_all_subvolumes_done(current_subvolumes, points_stack, labels_stack, colors_stack, normals_stack):
        return tf.greater_equal(total_subvolumes, current_subvolumes)

    def add_new_subvolume(current_subvolumes, points_stack, labels_stack, colors_stack, normals_stack):
        current_i = tf.reshape(tf.div(current_subvolumes, n_subvolume_y), [])
        current_j = tf.reshape(tf.mod(current_subvolumes, n_subvolume_y), [])
        current_min = coordmin + tf.convert_to_tensor(
            [tf.cast(current_i, tf.float32) * tf.constant(1.5), tf.cast(current_j, tf.float32) * 1.5, 0])
        current_max = coordmin + tf.convert_to_tensor(
            [(tf.cast(current_i, tf.float32) + 1) * 1.5, (tf.cast(current_j, tf.float32) + 1) * 1.5,
             tf.reshape(tf.gather(coordmax, [2]) - tf.gather(coordmin, [2]), [])])
        # find points in area
        in_area = tf.logical_and(tf.greater_equal(points, (current_min - 0.2)),
                                 tf.greater((current_max + 0.2), points))
        where = tf.where(tf.reduce_all(in_area, axis=1))
        current_choice = tf.reshape(tf.where(tf.reduce_all(in_area, axis=1)), [-1])
        # find current points
        cur_points = tf.gather(points, current_choice, name="6g")
        cur_labels = tf.gather(labels, current_choice, name="7g")
        cur_colors = tf.gather(colors, current_choice, name="8g")
        cur_normals = tf.gather(normals, current_choice, name="9g")
        cur_len = tf.reduce_sum(tf.ones_like(cur_points))
        in_area = tf.logical_and(tf.greater_equal(cur_points, (current_min - 0.01)),
                                 tf.greater((current_max + 0.01), cur_points))
        mask = tf.reduce_all(in_area, axis=1)
        mask_indices = tf.where(mask)
        # check if subset is valid
        vidx = tf.ceil((tf.gather(cur_points, mask_indices, name="10g") - current_min) / (current_max - current_min)
                       * tf.convert_to_tensor([31.0, 31.0, 62.0]))
        vidx_0 = tf.reshape(tf.gather(tf.transpose(vidx), 0, name="11g"), [-1])
        vidx_1 = tf.reshape(tf.gather(tf.transpose(vidx), 1, name="12g"), [-1])
        vidx_2 = tf.reshape(tf.gather(tf.transpose(vidx), 2, name="13g"), [-1])
        vidx, _ = tf.unique(vidx_0 * 31.0 * 62.0 + vidx_1 * 62 + vidx_2)
        vidx_len = tf.reduce_sum(tf.ones_like(vidx))
        labeled_points = tf.cast(tf.reduce_sum(tf.cast(tf.greater(cur_labels, 0), tf.int32)), tf.float32)
        isvalid = tf.logical_and(tf.greater_equal(tf.divide(labeled_points, cur_len), 0.7),
                                 tf.greater_equal(vidx_len / 31.0 / 31.0 / 62.0, 0.02))
        # get subset of correct length
        cur_len = tf.reduce_sum(tf.ones_like(cur_labels, dtype=tf.float32))
        choice = tf.cast(tf.random_uniform((npoints,), 0, cur_len), tf.int32)

        def go_on():
            # choice = np.random.choice(len(cur_labels), npoints, replace=True)
            current_points = tf.gather(cur_points, choice, name="14g")
            current_labels = tf.gather(cur_labels, choice, name="15g")
            current_normals = tf.gather(cur_normals, choice, name="16g")
            current_colors = tf.gather(cur_colors, choice, name="17g")
            current_mask = tf.gather(mask, choice, name="18g")
            current_sample_weight = tf.gather(label_weights, current_labels, name="19g")
            current_sample_weight *= tf.cast(current_mask, tf.float32)

            def concat():
                points_stack_new = tf.concat([points_stack, tf.expand_dims(current_points, 0)], 0)
                labels_stack_new = tf.concat([labels_stack, tf.expand_dims(current_labels, 0)], 0)
                colors_stack_new = tf.concat([colors_stack, tf.expand_dims(current_colors, 0)], 0)
                normals_stack_new = tf.concat([normals_stack, tf.expand_dims(current_normals, 0)], 0)
                return points_stack_new, labels_stack_new, colors_stack_new, normals_stack_new

            def keep():
                return points_stack, labels_stack, colors_stack, normals_stack

            points_stack_new, labels_stack_new, colors_stack_new, normals_stack_new = tf.cond(isvalid, concat, keep)
            return points_stack_new, labels_stack_new, colors_stack_new, normals_stack_new

        def skip():
            return points_stack, labels_stack, colors_stack, normals_stack

        # break if len == 0
        points_stack, labels_stack, colors_stack, normals_stack = \
            tf.cond(tf.greater(cur_len, tf.constant(0.0)), go_on, skip)
        return current_subvolumes + 1, points_stack, labels_stack, colors_stack, normals_stack

    tf.while_loop(not_all_subvolumes_done, add_new_subvolume,
                  loop_vars=[current_subvolumes, points_stack, labels_stack, colors_stack, normals_stack],
                  shape_invariants=[tf.TensorShape([]), tf.TensorShape([None, npoints, 3]),
                                    tf.TensorShape([None, npoints]), tf.TensorShape([None, npoints, 3]),
                                    tf.TensorShape([None, npoints, 3])])

    return points_stack, labels_stack, colors_stack, normals_stack, tf.ones_like(labels_stack)


def get_all_subsets_for_scene_numpy(points: np.ndarray, labels: np.ndarray, colors: np.ndarray, normals: np.ndarray,
                                    npoints: str = 8192) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    numpy function to get chunks with K points for every area of a scene
    this method is derived from Charles Qi's implementation

    :param points: (Nx3)
    :param labels: (N)
    :param colors: (Nx3)
    :param normals: (Nx3)
    :param npoints: (K) count of points to output (default 8192)

    :return: points_stack (X,K,3), labels_stack (X,K), colors_stack (X,K,3), normals_stack (X,K,3),
        sample_weight_stack(X,K)
    """
    label_weights = np.ones(21)
    label_weights[0] = 0

    coordmax = np.max(points, axis=0)
    coordmin = np.min(points, axis=0)
    nsubvolume_x = np.ceil((coordmax[0] - coordmin[0]) / 1.5).astype(np.int32)
    nsubvolume_y = np.ceil((coordmax[1] - coordmin[1]) / 1.5).astype(np.int32)
    point_sets = []
    semantic_segs = []
    sample_weights = []
    colors_sets = []
    normals_sets = []
    for i in range(nsubvolume_x):
        for j in range(nsubvolume_y):
            curmin = coordmin + [i * 1.5, j * 1.5, 0]
            curmax = coordmin + [(i + 1) * 1.5, (j + 1) * 1.5, coordmax[2] - coordmin[2]]
            curchoice = np.sum((points >= (curmin - 0.2)) * (points <= (curmax + 0.2)), axis=1) == 3
            cur_point_set = points[curchoice, :]
            cur_semantic_seg = labels[curchoice]
            cur_colors = colors[curchoice]
            cur_normals = normals[curchoice]
            if len(cur_semantic_seg) == 0:
                continue
            mask = np.sum((cur_point_set >= (curmin - 0.001)) * (cur_point_set <= (curmax + 0.001)), axis=1) == 3
            print("mask", mask.shape, type(mask), mask)
            choice = np.random.choice(len(cur_semantic_seg), npoints, replace=True)
            point_set = cur_point_set[choice]  # Nx3
            normal_cur = cur_normals[choice]
            color_cur = cur_colors[choice]
            semantic_seg = cur_semantic_seg[choice]  # N
            mask = mask[choice]
            if sum(mask) / float(len(mask)) < 0.01:
                continue
            sample_weight = label_weights[semantic_seg]
            sample_weight *= mask  # N
            point_sets.append(np.expand_dims(point_set, 0))  # 1xNx3
            semantic_segs.append(np.expand_dims(semantic_seg, 0))  # 1xN
            sample_weights.append(np.expand_dims(sample_weight, 0))  # 1xN
            colors_sets.append(np.expand_dims(color_cur, 0))
            normals_sets.append(np.expand_dims(normal_cur, 0))
    point_sets = np.concatenate(tuple(point_sets), axis=0)
    semantic_segs = np.concatenate(tuple(semantic_segs), axis=0)
    sample_weights = np.concatenate(tuple(sample_weights), axis=0)
    colors_sets = np.concatenate(tuple(colors_sets), axis=0)
    normals_sets = np.concatenate(tuple(normals_sets), axis=0)
    return point_sets, semantic_segs, colors_sets, normals_sets, sample_weights


def random_rotate(points: tf.Tensor, labels: tf.Tensor, colors: tf.Tensor, normals: tf.Tensor,
                  sample_weight: tf.Tensor) \
        -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    tensorflow function to randomly rotate point cloud

    :param points: (Nx3)
    :param labels: (N)
    :param colors: (Nx3)
    :param normals: (Nx3)
    :param sample_weight: (N)
    :return: rotated points(Nx3), labels(N), colors(Nx3), rotated normals(Nx3), sample_weight: (N)
    """
    alpha = tf.random_uniform([], 0, math.pi * 2)
    rot_matrix = tf.convert_to_tensor(
        [[tf.cos(alpha), -tf.sin(alpha), 0], [tf.sin(alpha), tf.cos(alpha), 0], [0, 0, 1]])
    rot_points = tf.matmul(points, rot_matrix)
    rot_normals = tf.matmul(normals, rot_matrix)
    return rot_points, labels, colors, rot_normals, sample_weight


def get_transformed_dataset(train: str, prefetch: bool = True, threads: int = 4):
    """
    tensorflow dataset, to load and transform files asynchronous

    :param train: one of  {"train", "val", "train_subset"}
    :param prefetch: prefetches data if True
    :param threads: number of parallel threads to use
    :return:
    """
    ds = gd.get_dataset(train)
    if prefetch:
        # prefetch loading from disk
        ds = ds.prefetch(threads)
    ds = ds.map(label_map, threads)
    if train == "train" or train == "train_subset":
        ds = ds.map(get_subset, threads)
        ds = ds.map(random_rotate, threads)
    elif train == "val":
        ds = ds.map(get_all_subsets_for_scene, threads)
    else:
        raise ValueError("train must be one of  {'train', 'val', 'train_subset'}")
    return ds

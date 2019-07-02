import math

import numpy as np
import tensorflow as tf

import scannet_dataset.generator_dataset as gd


def label_map(points, labels, colors, normals):
    map = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 12, 13: 13, 15: 14, 23: 15, 27: 16,
           32: 17, 33: 18, 35: 19, 38: 20}
    map = tf.convert_to_tensor([map.get(i, 0) for i in range(41)])
    labels = tf.minimum(labels, 40)
    mapped_labels = tf.gather(map, labels)
    # mapped_labels = [map.get(label, 0) for label in labels]
    # mapped_labels = tf.convert_to_tensor(mapped_labels)
    return points, mapped_labels, colors, normals


def get_subset(points, labels, colors, normals):
    npoints = 8192
    label_weights = tf.ones(21)  # TODO set weights accordingly

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


def random_rotate(points, labels, colors, normals, sample_weight):
    alpha = tf.random_uniform([], 0, math.pi * 2)
    rot_matrix = tf.convert_to_tensor(
        [[tf.cos(alpha), -tf.sin(alpha), 0], [tf.sin(alpha), tf.cos(alpha), 0], [0, 0, 1]])
    rot_points = tf.matmul(points, rot_matrix)
    rot_normals = tf.matmul(normals, rot_matrix)
    return rot_points, labels, colors, rot_normals, sample_weight




def get_transformed_dataset(train):
    ds = gd.get_dataset(train)
    ds = ds.map(label_map, 4)
    ds = ds.map(get_subset, 4)
    ds = ds.map(random_rotate, 4)
    return ds


if __name__ == '__main__':
    ds = gd.get_dataset("train")
    ds = ds.map(label_map, 4)
    ds = ds.map(get_subset, 4)
    ds = ds.map(random_rotate, 4)
    ds = ds.prefetch(64)
    points, labels, colors, normals, sample_weight = ds.make_one_shot_iterator().get_next()
    norm_sizes = tf.norm(normals, axis=1)
    sess = tf.Session()
    val = sess.run(points)
    print(type(val))
    print("hello")
    np.save("/home/tim/test.npy", val)
    print("good bye")

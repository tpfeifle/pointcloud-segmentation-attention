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
    # label_weights = tf.ones(21)  # TODO verify following weights
    label_weights = tf.constant([0, 3.8175557, 2.2230785, 2.6964862, 4.546552, 4.9208603, 5.0999, 4.9115996,
                                 5.02148, 4.9090133, 5.4020867, 5.401546, 5.4178405, 5.1401854, 5.332984,
                                 4.9614744, 5.259515, 5.439167, 5.3803735, 5.393622, 4.909173, 4.9360685])

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


def get_all_subsets_for_scene(points, labels, colors, normals):
    npoints = 8192
    label_weights = tf.ones(21)  # TODO set weights accordingly

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


def random_rotate(points, labels, colors, normals, sample_weight):
    alpha = tf.random_uniform([], 0, math.pi * 2)
    rot_matrix = tf.convert_to_tensor(
        [[tf.cos(alpha), -tf.sin(alpha), 0], [tf.sin(alpha), tf.cos(alpha), 0], [0, 0, 1]])
    rot_points = tf.matmul(points, rot_matrix)
    rot_normals = tf.matmul(normals, rot_matrix)
    return rot_points, labels, colors, rot_normals, sample_weight


def get_transformed_dataset(train, prefetch=True):
    ds = gd.get_dataset(train)
    if prefetch:
        # prefetch loading from disk
        ds = ds.prefetch(8)
    ds = ds.map(label_map, 4)
    if train == "train":
        ds = ds.map(get_subset, 4)
        ds = ds.map(random_rotate, 4)
    elif train == "val":
        ds = ds.map(get_all_subsets_for_scene, 4)
    else:
        raise ValueError("train must be either 'train' or 'val'")
    return ds


if __name__ == '__main__':
    ds = gd.get_dataset("val")
    ds = ds.map(label_map, 4)
    ds = ds.map(get_all_subsets_for_scene, 4)
    points, labels, colors, normals, sample_weight = ds.make_one_shot_iterator().get_next()
    sess = tf.Session()
    val = sess.run(points)
    print(type(val))
    print(val.shape)
    print("hello")
    exit(0)
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

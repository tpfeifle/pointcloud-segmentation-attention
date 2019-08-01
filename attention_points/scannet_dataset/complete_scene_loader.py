import numpy as np


def get_all_subsets_with_all_points_for_scene_features(points, features, get_sample_weights):
    """
    numpy function to get all points of a scene grouped by chunks
    this method can be used to get values for all points of a scene e.g. the test set
    it also returns the original indices of all samples to map values or predictions back to original points
    :return: point_sets, feature sets, masks_sets, points_orig_idxs_sets
    """
    npoints = 8192
    label_weights = np.ones(21)
    label_weights[0] = 0
    points_orig_idxs = np.arange(len(points), dtype=int)

    def shuffle_forward(l):
        order = list(range(len(l)))
        np.random.shuffle(order)
        return list(np.array(l)[order]), order

    coordmax = np.max(points, axis=0)
    coordmin = np.min(points, axis=0)
    nsubvolume_x = np.ceil((coordmax[0] - coordmin[0]) / 1.5).astype(np.int32)
    nsubvolume_y = np.ceil((coordmax[1] - coordmin[1]) / 1.5).astype(np.int32)
    point_sets = []
    feature_sets = [[] for feature in features]
    sample_weights = []
    masks_sets = []
    points_orig_idxs_sets = []
    mask_counter = 0
    for i in range(nsubvolume_x):
        for j in range(nsubvolume_y):
            curmin = coordmin + [i * 1.5, j * 1.5, 0]
            curmax = coordmin + [(i + 1) * 1.5, (j + 1) * 1.5, coordmax[2] - coordmin[2]]
            curchoice = np.sum((points >= (curmin - 0.2)) * (points <= (curmax + 0.2)), axis=1) == 3
            cur_point_set = points[curchoice]
            cur_features = [feature[curchoice] for feature in features]
            cur_points_orig_idxs = points_orig_idxs[curchoice]
            if len(cur_point_set) == 0:
                # print('empty subset')
                continue
            mask = np.sum((cur_point_set >= curmin) * (cur_point_set <= curmax), axis=1) == 3
            mask_counter += np.sum(mask)
            # choice = np.random.choice(len(cur_semantic_seg), npoints, replace=True)
            # 1. Shuffle points in cur_point_set and keep mapping to original
            cur_point_set, order_to_inverse_shuffle = shuffle_forward(cur_point_set)
            cur_features = [feature[order_to_inverse_shuffle] for feature in cur_features]
            mask = mask[order_to_inverse_shuffle]
            cur_points_orig_idxs = cur_points_orig_idxs[order_to_inverse_shuffle]
            for feature in cur_features:
                assert len(cur_point_set) == len(feature)
            assert len(cur_point_set) == len(mask)
            assert len(cur_point_set) == len(cur_points_orig_idxs)
            # print("number of unique point ids %s" % len(np.unique(cur_points_orig_idxs)))

            k = 0
            for k in range(int(len(cur_point_set) / npoints)):
                offset = k * npoints
                point_set = cur_point_set[offset:offset + npoints]
                features_cur = [feature[offset:offset + npoints] for feature in cur_features]
                mask_cur = mask[offset:offset + npoints]
                points_orig_idxs_cur = cur_points_orig_idxs[offset:offset + npoints]
                if sum(mask_cur) == 0:  # TODO: why is len(mask) often Zero? --> remove the len(mask) == 0
                    # print("oh no aaa!!")
                    continue
                if get_sample_weights:
                    sample_weight = label_weights[features_cur[0]]
                else:
                    sample_weight = np.ones(len(point_set))
                sample_weight *= mask_cur  # N
                point_sets.append(np.expand_dims(point_set, 0))  # 1xNx3
                for feature_index, feature_set in enumerate(feature_sets):
                    feature_set.append(np.expand_dims(features_cur[feature_index], 0))
                sample_weights.append(np.expand_dims(sample_weight, 0))
                masks_sets.append(
                    np.expand_dims(mask_cur, 0))  # needed to ignore predictions of points outside of the actual cube
                points_orig_idxs_sets.append(np.expand_dims(points_orig_idxs_cur, 0))

            rest_idxs = len(cur_point_set) % npoints

            ### Only for the rest all again
            if len(cur_point_set) > npoints:
                # print("rest %s " % rest_idxs)
                k = k + 1
            offset = k * npoints
            # add random points of this "subset-frame" to fill up to npoints (predictions for them get removed through masking)
            fill_up_idxs = np.random.choice(len(cur_point_set), npoints - rest_idxs, replace=True)
            # fill_up_idxs = np.ones(npoints-rest_idxs, dtype=np.int32)
            point_set = np.concatenate(
                (cur_point_set[offset:offset + rest_idxs], np.array(cur_point_set)[fill_up_idxs]))
            features_cur = [np.concatenate((feature[offset:offset + rest_idxs], feature[fill_up_idxs]))
                            for feature in cur_features]
            # filter the added points out when saving predictions (so make them zero in the mask)
            mask_cur = np.concatenate((mask[offset:offset + rest_idxs], np.zeros(npoints - rest_idxs, dtype=bool)))
            points_orig_idxs_cur = np.concatenate(
                (cur_points_orig_idxs[offset:offset + rest_idxs], np.zeros(npoints - rest_idxs, dtype=int)))
            if sum(mask_cur) == 0:
                continue
            if get_sample_weights:
                sample_weight = label_weights[features_cur[0]]
            else:
                sample_weight = np.ones(len(point_set))
            point_sets.append(np.expand_dims(point_set, 0))  # 1xNx3
            sample_weights.append(np.expand_dims(sample_weight, 0))  # 1xN
            for feature_index, feature_set in enumerate(feature_sets):
                feature_set.append(np.expand_dims(features_cur[feature_index], 0))
            masks_sets.append(
                np.expand_dims(mask_cur, 0))  # needed to ignore predictions of points outside of the actual cube
            points_orig_idxs_sets.append(np.expand_dims(points_orig_idxs_cur, 0))
            ### END: Only for the rest all again

    point_sets = np.concatenate(tuple(point_sets), axis=0)
    sample_weights = np.concatenate(tuple(sample_weights), axis=0)
    feature_sets = [np.concatenate(tuple(feature)) for feature in feature_sets]
    masks_sets = np.concatenate(tuple(masks_sets), axis=0)
    points_orig_idxs_sets = np.concatenate(tuple(points_orig_idxs_sets), axis=0)
    return point_sets, feature_sets, sample_weights, masks_sets, points_orig_idxs_sets


def get_all_subsets_with_all_points_for_scene_numpy(points, labels, colors, normals):
    point_sets, feature_sets, sample_weights, masks_sets, points_orig_idxs_sets = \
        get_all_subsets_with_all_points_for_scene_features(points, [labels, colors, normals], True)
    return point_sets, feature_sets[0], feature_sets[1], feature_sets[2], \
           sample_weights, masks_sets, points_orig_idxs_sets


def get_all_subsets_with_all_points_for_scene_numpy_test(points, colors, normals):
    point_sets, feature_sets, sample_weights, masks_sets, points_orig_idxs_sets = \
        get_all_subsets_with_all_points_for_scene_features(points, [colors, normals], False)
    return point_sets, feature_sets[0], feature_sets[1], masks_sets, points_orig_idxs_sets

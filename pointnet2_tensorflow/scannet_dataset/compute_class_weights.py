import numpy as np

from scannet_dataset import generator_dataset

label_map = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12, 14: 13, 16: 14, 24: 15,
             28: 16, 33: 17, 34: 18, 36: 19, 39: 20}


def get_weights():
    gen_train = generator_dataset.tf_train_generator()
    bin_count = np.zeros(21, dtype=int)
    for i in range(1201):
        points, labels, colors, normals = gen_train.__next__()
        labels = [label_map.get(i, 0) for i in labels]
        add_value = np.zeros(21, dtype=int)
        current_bin_count = np.bincount(labels)
        add_value[:current_bin_count.shape[0]] = current_bin_count
        bin_count += add_value
        if (i + 1) % 100 == 0:
            print(f"finished {i + 1} scenes")
            print(bin_count)

    print(bin_count)
    total = np.sum(bin_count)
    weights = total / (len(bin_count) * bin_count)
    return weights


if __name__ == '__main__':
    counts = [43590149, 41822096, 31929944, 5646791, 3762480, 9929883, 3401149, 4921067, 6294926, 5426047, 3292834,
              678377, 667652, 2675491, 3012156, 721874, 437510, 435576, 359104, 475034, 4869969]
    counts = np.array(counts)
    w = np.sum(counts) / (len(counts) * counts)
    # w = get_weights()
    print(w)
    weights = [0.19046473, 0.19851674, 0.26001881, 1.47028394, 2.20662599, 0.8361011, 2.44105334, 1.68711097,
               1.31890131, 1.53009846, 2.52134974, 12.23860205, 12.43519999, 3.10312617, 2.75629355, 11.50115691,
               18.97644886, 19.06070615, 23.11972616, 17.47745665, 1.70481293]

# total bincount of train set is:
# [43590149, 41822096, 31929944, 5646791, 3762480, 9929883, 3401149, 4921067, 6294926, 5426047, 3292834,
#  678377, 667652, 2675491, 3012156, 721874, 437510, 435576, 359104, 475034, 4869969]
# weights from this are:
# [0.19046473, 0.19851674, 0.26001881, 1.47028394, 2.20662599, 0.8361011, 2.44105334, 1.68711097,
#                1.31890131, 1.53009846, 2.52134974, 12.23860205, 12.43519999, 3.10312617, 2.75629355, 11.50115691,
#                18.97644886, 19.06070615, 23.11972616, 17.47745665, 1.70481293]

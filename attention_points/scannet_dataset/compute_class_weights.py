"""
This module can analyze the distribution of class labels and compute weights accordingly
The formula 1 / log(1.2 + counts / sum(counts)) is adapted from Charles Qi's implementation
"""
import numpy as np

from attention_points.scannet_dataset import generator_dataset

LABEL_MAP = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12, 14: 13, 16: 14, 24: 15,
             28: 16, 33: 17, 34: 18, 36: 19, 39: 20}


def get_weights() -> np.ndarray:
    """
    computes the weights for classes to use for training
    note that class 0 is unannotated and should be weighted with 0

    :return: weights (21)
    """
    gen_train = generator_dataset.tf_train_generator()
    bin_count = np.zeros(21, dtype=int)
    for i in range(1201):
        points, labels, colors, normals = gen_train.__next__()
        labels = [LABEL_MAP.get(i, 0) for i in labels]
        add_value = np.zeros(21, dtype=int)
        current_bin_count = np.bincount(labels)
        add_value[:current_bin_count.shape[0]] = current_bin_count
        bin_count += add_value
        if (i + 1) % 100 == 0:
            print(f"finished {i + 1} scenes")
            print(bin_count)

    print(bin_count)
    total = np.sum(bin_count)
    weights = 1 / np.log(1.2 + bin_count / total)
    return weights


if __name__ == '__main__':
    # weights from previously computed counts
    counts = [43590149, 41822096, 31929944, 5646791, 3762480, 9929883, 3401149, 4921067, 6294926, 5426047, 3292834,
              678377, 667652, 2675491, 3012156, 721874, 437510, 435576, 359104, 475034, 4869969]
    counts = np.array(counts)
    w = 1 / np.log(1.2 + counts / np.sum(counts))
    print(list(w))
    # count new
    w = get_weights()
    print(list(w))

# total bincount of train set is:
# [43590149, 41822096, 31929944, 5646791, 3762480, 9929883, 3401149, 4921067, 6294926, 5426047, 3292834,
#  678377, 667652, 2675491, 3012156, 721874, 437510, 435576, 359104, 475034, 4869969]
# weights from this are:
#   [2.6912544922435333, 2.743064592944318, 3.0830506790927132, 4.785754459526457, 4.9963745147506184,
#    4.372710774561782, 5.039124880965811, 4.86451825464344, 4.717751595568025, 4.809412839311939,
#    5.052097251455304, 5.389129668645318, 5.390614085649042, 5.127458225110977, 5.086056870814752,
#    5.3831185190895265, 5.422684124268539, 5.422955391988761, 5.433705358072363, 5.417426773812747,
#    4.870172044153657]

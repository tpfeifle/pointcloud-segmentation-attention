"""
Extracts training scores from the tensorboard logs and plots them with matplotlib.

Plotted Models:
- Attention Models (using attention in layer X)
- PointNet++ with Additional Features
"""

import matplotlib.pyplot as plt
import tensorflow as tf

log_path = '/home/tim/training_log/'
DEBUG_MODE = False


def plot_scores_attention_models(ylabel="mIoU", tag='iou'):
    """
    Reads the tensorboard logs for the `attention-models` from disk and plots the specified metric

    :param ylabel: label for the y-axis
    :param tag: element of ['loss', 'accuracy', iou'] that should be plotted
    :return:
    """
    event_files = [log_path + 'subset/baseline1563868622_val/events.out.tfevents.1563868758.tim-desktop',
                   log_path + 'subset/attention_in_layer_0_1563879995_val/events.out.tfevents.1563880135.tim-desktop',
                   log_path + 'subset/attention_in_layer_1_1563898776_val/events.out.tfevents.1563898912.tim-desktop',
                   log_path + 'subset/attention_in_layer_2_1563909261_val/events.out.tfevents.1563909397.tim-desktop',
                   log_path + 'subset/attention_in_layer_3_1563919768_val/events.out.tfevents.1563919906.tim-desktop']
    titles = ['Baseline',
              'Att-Layer-1',
              'Att-Layer-2',
              'Att-Layer-3',
              'Att-Layer-4']
    for idx, path_to_events_file in enumerate(event_files):

        loss = []
        accuracy = []
        iou = []

        for e in tf.train.summary_iterator(path_to_events_file):
            for v in e.summary.value:
                if v.tag == 'loss':
                    loss.append(v.simple_value)
                    DEBUG_MODE and print("loss: %s" % v.simple_value)
                if v.tag == 'accuracy':
                    accuracy.append(v.simple_value)
                    DEBUG_MODE and print("Accuracy: %s" % v.simple_value)
                if v.tag == 'iou':
                    iou.append(v.simple_value)
                    DEBUG_MODE and print("Iou: %s" % v.simple_value)

        epoch = range(0, len(loss) * 4, 4)
        title = titles[idx]

        if tag == 'iou':
            plt.plot(epoch, iou, label=title)
        elif tag == 'accuracy':
            plt.plot(epoch, accuracy, label=title)
        else:
            plt.plot(epoch, loss, label=title)

        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.legend(loc='upper left')
        plt.title("Validation mIoU on Subset")

    plt.savefig("validation_iou.svg")
    plt.show()


def plot_scores_feature_model():
    """
    Reads the tensorboard logs for the baseline (PointNet++) and the model with additional features from disk
    and plots the iou metric

    :return:
    """
    iou = []
    for e in tf.train.summary_iterator(
            log_path + '/baseline/working_baseline_250epochs_val/events.out.tfevents.1563534342.tim-desktop'):
        for v in e.summary.value:
            if v.tag == 'iou':
                iou.append(v.simple_value)
    epoch = range(0, len(iou) * 4, 4)
    title = "PointNet++$^1$"
    plt.plot(epoch, iou, label=title)

    # With Features consisted of a first and a continued run
    iou = []

    # Run1
    for e in tf.train.summary_iterator(
            log_path + '/pointnet_and_features/long_run1563786310_val/events.out.tfevents.1563786767.tim-desktop'):
        for v in e.summary.value:
            if v.tag == 'iou':
                iou.append(v.simple_value)
    # Run2
    skip_first = 2
    for e in tf.train.summary_iterator(
            log_path + '/pointnet_and_features/long_run1563786310_continued_val/'
                       'events.out.tfevents.1563815651.tim-desktop'):
        for v in e.summary.value:
            if v.tag == 'iou' and skip_first < 0:
                iou.append(v.simple_value)
            skip_first -= 1
    epoch = range(0, len(iou) * 4, 4)
    title = "Additional Features (ours)"
    plt.plot(epoch[:65], iou[:65], label=title)

    plt.xlabel("Epoch")
    plt.ylabel("mIoU")
    plt.legend(loc='upper left')
    plt.title("Validation mIoU")
    plt.savefig("validation_iou_no_attention.svg")
    plt.show()


if __name__ == '__main__':
    plot_scores_feature_model()
    plot_scores_attention_models()

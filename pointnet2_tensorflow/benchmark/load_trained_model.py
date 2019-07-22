import numpy as np
import tensorflow as tf
import models.pointnet2_sem_seg_features as model
from scannet_dataset import generator_dataset


N_POINTS = 8192
N_VAL_SAMPLES = 4542
BATCH_SIZE = 1


def normalize_features_fixed(x, current_range):
    current_min, current_max = current_range[0], current_range[1]
    normed_min, normed_max = 0, 3
    x_normed = (x - current_min) / (current_max - current_min)
    x_normed = x_normed * (normed_max - normed_min) + normed_min
    return x_normed

def train(batch_size=BATCH_SIZE):
    tf.Graph().as_default()
    # tf.device('/gpu:0')
    tf.reset_default_graph()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    # define validation data
    val_data = generator_dataset.get_dataset("eval")
    val_data = val_data.batch(batch_size).prefetch(4)
    val_iterator = tf.data.Iterator.from_structure(val_data.output_types, val_data.output_shapes)
    val_data_init = val_iterator.make_initializer(val_data)
    sess.run(val_data_init)
    points, labels, colors, normals, scene_name = val_iterator.get_next()
    # val_features = tf.concat([tf.cast(colors, tf.float32), normals], 2)
    val_features = tf.concat([normalize_features_fixed(tf.cast(colors, tf.float32), [0, 255]),
                                      normalize_features_fixed(normals, [-1, 1])], 2)
    val_coordinates = points
    val_labels = labels


    # define model and metrics
    is_training_pl = tf.Variable(False)

    # validation metrics
    val_pred, _ = model.get_model(val_coordinates, val_features, is_training_pl, 21)

    # Filter out the unassigned labels
    val_labels_flat = tf.reshape(val_labels, [-1])
    val_pred_flat = tf.reshape(val_pred, [-1, val_pred.shape[2]])

    loc = tf.reshape(tf.where(val_labels_flat > 0), [-1])
    val_labels_assigned = tf.gather(val_labels_flat, loc)
    val_pred_assigned = tf.gather(val_pred_flat, loc)
    correct_val_pred = tf.equal(tf.argmax(val_pred_assigned, 1, output_type=tf.int32), val_labels_assigned)

    val_acc = tf.reduce_sum(tf.cast(correct_val_pred, tf.float32)) / \
              tf.cast(tf.shape(val_labels_assigned)[0], tf.float32)
    val_iou, val_iou_update = tf.metrics.mean_iou(val_labels_assigned,
                                                  tf.argmax(val_pred_assigned, 1, output_type=tf.int32),
                                                  num_classes=21, name="val_iou")
    # initialize variables
    variable_init = tf.global_variables_initializer()

    sess.run(variable_init)
    sess.run(tf.local_variables_initializer())
    saver = tf.train.Saver()
    get_checkpoint = tf.train.latest_checkpoint("/home/tim/training_log/baseline/features_1563618425_train")
    saver.restore(sess, get_checkpoint)

    val_batches = N_VAL_SAMPLES // BATCH_SIZE
    print(f"starting evaluation {val_batches} batches")
    all_pred = []
    all_labels = []
    all_points = []
    current_scene = ""
    for j in range(val_batches):
        predictions_res, labels_res, points_res, scene, thefeatures, acc_train, train_iou_val = sess.run([val_pred, val_labels, val_coordinates, scene_name, val_features, val_acc, val_iou])
        max_pred = np.argmax(predictions_res, axis=2)
        print(f"\taccuracy: {acc_train:.4f}\taccumulated iou: {train_iou_val:.4f}")
        if scene != current_scene and j != 0:
            print("got all for one scene so please evaluate this scene")
            all_points = np.concatenate(all_points)
            all_pred = np.concatenate(all_pred)
            # pc_util.draw_point_cloud(all_points)
            np.save("/home/tim/features_points_%s.npy" % scene[0].decode('ascii'), all_points)
            np.save("/home/tim/features_pred_%s.npy" % scene[0].decode('ascii'), all_pred)

            #image1 = pc_util.point_cloud_three_views(all_points)
            #img = Image.fromarray(np.uint8(image1 * 255.0))
            #img.save(scene[0].decode('ascii') + "_3views.jpg")
            #image = pc_util.point_cloud_to_image(all_points, 500)
            #img = Image.fromarray(np.uint8(image * 255.0))
            #img.save(scene[0].decode('ascii') + "toImage.jpg")
            #uvidx, uvlabel, nvox = pc_util.point_cloud_label_to_surface_voxel_label(all_points, all_pred)  # res=

            all_pred = []
            all_labels = []
            all_points = []
        all_pred.append(np.squeeze(max_pred))
        all_labels.append(np.squeeze(val_labels))
        all_points.append(np.squeeze(points_res))
        current_scene = scene



if __name__ == '__main__':
    train()

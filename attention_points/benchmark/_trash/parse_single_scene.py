import pickle
import os

data_filename = os.path.join('/home/tim/data/parts/to_process.pickle')
output = os.path.join('/home/tim/data/parts/single_scene.pickle')

with open(data_filename, 'rb') as fp:
    single_scene_points = [pickle.load(fp, encoding='latin1')[0]]
    single_scene_labels = [pickle.load(fp, encoding='latin1')[0]]
    with open(output, 'wb') as wfp:
        pickle.dump(single_scene_points, wfp)
        pickle.dump(single_scene_labels, wfp)

# Attention Points
This project uses an attention mechanism to replace the max pooling operation of PointNet++.
As a dataset we used ScanNetv2.
It also contains a model which can use more features than the reference implementation,
i.e. color features and surface normal vectors of points.

### Structure
`pointnet2_tensorflow` contains an almost untouched version of the reference implementation of PointNet++ by Charles Qi.

`pytorch_implementation` contains an alternative implementation of PointNet++ written in Pytorch by [halimacc https://github.com/halimacc/pointnet3].

`Attention_Points ` contains our additional code, including a new data pipeline, model variations,
training methods, visualizations and a new way create predictions for large point clouds.


### Dataset

# Running the models
### Models
### Training Methods


## Output
### Benchmark
To predict labels for each point in each scene `generate_predictions.py` takes as input a trained model.
We first create subsets for each scene (see `scannet_dataset/complete_scene_loader.py`), each containing random 8192 points
and then keep the predictions for our inner cuboid region by masking out the others and inversing our shuffling.
The predictions are then stored in two different formats:
1. For visualization with the visualization scripts as numpy-arrays
2. For evaluation on the benchmark in the benchmark format (one label per line in files following the naming `scene%04d_%02d.txt`)

We evaluated our model using the additional features using the official [ScanNet-Benchmark](http://kaldir.vc.in.tum.de/scannet_benchmark/).
The validation benchmark scores can be calculated using the additional scripts in the benchmark folder.
Those contain the `generate_groundtruth.py` that generates the `scene%04d_%02d.txt` files with the correct labels
and the `evaluate.py` script that compares the predicted labels for each scene with the groundtruth and outputs the IoU by class 
as well as the confusion matrix. 


### Visualization
The predicted labels can also be qualitatively evaluated. The script `qualitative_animations.py` takes 
the points and predicted labels of scenes as input and visualizes them using the pptk-viewer.

!["Example frame of scene0660_00"](attention_points/visualization/examples/frame_079.png?raw=true "Frame of scene 0660_00")	


It rotates the scenes and saves those animation frames as images. Those can be converted into videos using
e.g. ffmpeg with:
`ffmpeg -i "scene0XXX_0X/frame_%03d.png" -c:v mpeg4 -qscale:v 0 -r 24 scene0XXX_0X.mp4`

If one wants to debug the performance of the model during training one can instead use the `labels_during_training.py` file to animate
the predicted labels over the time of multiple training steps. 
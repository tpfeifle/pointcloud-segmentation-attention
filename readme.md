Attention Points
================
This project uses an attention mechanism to replace the max pooling operation of PointNet++.
As a dataset we used ScanNetv2.
It also contains a model which can use more features than the reference implementation,
i.e. color features and surface normal vectors of points.

Structure
---------
`pointnet2_tensorflow` contains an almost untouched version of the reference implementation of PointNet++ by Charles Qi.

`pytorch_implementation` contains an alternative implementation of PointNet++ written in Pytorch by [halimacc https://github.com/halimacc/pointnet3].

`Attention_Points ` contains our additional code, including a new data pipeline, model variations,
training methods, visualizations and a new way create predictions for large point clouds.

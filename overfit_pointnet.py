"""
copied from https://github.com/halimacc/pointnet3
"""
from pointnet2 import *
import torch
import torch.optim as optim
import torch.nn.functional as F
from data_loader import ScanNetDatasetWrapper


def train():
    # init training dataset
    train_dataset = ScanNetDatasetWrapper(train=True)

    classifier = PointNet2SemSeg(num_classes=40)
    lr = 1e-2
    optimizer = optim.Adam(classifier.parameters(), lr=lr)

    # define loss function
    loss_fun = torch.nn.CrossEntropyLoss()

    print("Start training...")
    classifier.train()

    data = train_dataset[44]
    pointcloud, label = data
    pointcloud, label = pointcloud.unsqueeze(0), label.unsqueeze(0)
    print(pointcloud.shape, label.shape)  # should be [1, N ,3] [1, N]
    print(f"number of points: {label.shape[1]}")
    for i in range(100):
        pointcloud, label = data
        pointcloud, label = pointcloud.unsqueeze(0), label.unsqueeze(0)
        pointcloud = pointcloud.permute(0, 2, 1)
        # TODO activate cuda again
        # pointcloud, label = pointcloud.cuda(), label.cuda()

        optimizer.zero_grad()
        pred = classifier(pointcloud)

        loss = loss_fun(pred, label)
        pred_choice = pred.max(1)[1]

        loss.backward()
        optimizer.step()
        lr *= 0.95
        optimizer = optim.Adam(classifier.parameters(), lr=lr)

        print(f"loss: {loss.item()}")
        current_correct_examples = pred_choice.eq(label.view(-1)).sum().item()

        print(f"correct examples: {current_correct_examples}")
        print(f"current accuracy: {100.0 * current_correct_examples / label.shape[1]}")


if __name__ == '__main__':
    train()

"""Trains a LeNet-5 classifier for MNIST.

Adapted from "Training a Classifier," a tutorial in the PyTorch 60-minute blitz:
https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py

Requirements:
    torch==1.7
    torchvision==0.8

Usage:
    # Saves classifier to mnist_classifier.pth (We have saved a copy in
    # tutorials/mnist/).
    python train_mnist_classifier.py

    # Evaluates an existing network.
    python train_mnist_classifier.py FILE.pth
"""
import sys

import torch
import torch.nn as nn
import torchvision

MEAN_TRANSFORM = 0.1307
STD_DEV_TRANSFORM = 0.3081
TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE = 1000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fit(net, epochs, trainloader):
    """Trains net for the given number of epochs."""
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(net.parameters())

    for epoch in range(epochs):
        print(f"=== Epoch {epoch + 1} ===")
        total_loss = 0.0

        # Iterate through batches in the shuffled training dataset.
        for batch_i, data in enumerate(trainloader):
            inputs = data[0].to(device)
            labels = data[1].to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if (batch_i + 1) % 100 == 0:
                print(f"Batch {batch_i + 1:5d}: {total_loss}")
                total_loss = 0.0


def evaluate(net, loader):
    """Evaluates the network's accuracy on the images in the dataloader."""
    correct_per_num = [0 for _ in range(10)]
    total_per_num = [0 for _ in range(10)]

    with torch.no_grad():
        for data in loader:
            images, labels = data
            outputs = net(images.to(device))
            _, predicted = torch.max(outputs.to("cpu"), 1)
            correct = (predicted == labels).squeeze()
            for label, c in zip(labels, correct):
                correct_per_num[label] += c.item()
                total_per_num[label] += 1

    for i in range(10):
        print(f"Class {i}: {correct_per_num[i] / total_per_num[i]:5.3f}"
              f" ({correct_per_num[i]} / {total_per_num[i]})")
    print(f"TOTAL  : {sum(correct_per_num) / sum(total_per_num):5.3f}"
          f" ({sum(correct_per_num)} / {sum(total_per_num)})")


def main():
    """Trains and saves the classifier."""
    print("Device:", device)

    # Transform each image by turning it into a tensor and then
    # normalizing the values.
    mnist_transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((MEAN_TRANSFORM,),
                                         (STD_DEV_TRANSFORM,))
    ])
    trainloader = torch.utils.data.DataLoader(torchvision.datasets.MNIST(
        './data', train=True, download=True, transform=mnist_transforms),
                                              batch_size=TRAIN_BATCH_SIZE,
                                              shuffle=True)
    testloader = torch.utils.data.DataLoader(torchvision.datasets.MNIST(
        './data', train=False, transform=mnist_transforms),
                                             batch_size=TEST_BATCH_SIZE,
                                             shuffle=False)

    lenet5 = nn.Sequential(
        nn.Conv2d(1, 6, (5, 5), stride=1, padding=0),  # (1,28,28) -> (6,24,24)
        nn.MaxPool2d(2),  # (6,24,24) -> (6,12,12)
        nn.ReLU(),
        nn.Conv2d(6, 16, (5, 5), stride=1, padding=0),  # (6,12,12) -> (16,8,8)
        nn.MaxPool2d(2),  # (16,8,8) -> (16,4,4)
        nn.ReLU(),
        nn.Flatten(),  # (16,4,4) -> (256,)
        nn.Linear(256, 120),  # (256,) -> (120,)
        nn.ReLU(),
        nn.Linear(120, 84),  # (120,) -> (84,)
        nn.ReLU(),
        nn.Linear(84, 10),  # (84,) -> (10,)
        nn.LogSoftmax(dim=1),  # (10,) log probabilities
    ).to(device)

    if len(sys.argv) > 1:
        print("===== Loading existing network for evaluation =====")
        filename = sys.argv[1]
        print("Filename:", filename)
        lenet5.load_state_dict(torch.load(filename, map_location=device))
    else:
        print("===== Fitting Network =====")
        fit(lenet5, 2, trainloader)

    print("===== Evaluation =====")
    print("=== Training Set Evaluation ===")
    evaluate(lenet5, trainloader)
    print("=== Test Set Evaluation ===")
    evaluate(lenet5, testloader)

    print("===== Saving Network =====")
    torch.save(lenet5.state_dict(), "mnist_classifier.pth")


if __name__ == "__main__":
    main()

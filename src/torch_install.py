from os import path
import torch
import torchvision
# set base_path
base_path = "./"
# install mnist,svhn,cifar10,cifar100
torchvision.datasets.MNIST(path.join(base_path,"dataset","mnist"),download=True)
torchvision.datasets.CIFAR10(path.join(base_path,"dataset","cifar"),download=True)
torchvision.datasets.CIFAR100(path.join(base_path,"dataset","cifar"),download=True)
torchvision.datasets.SVHN(path.join(base_path,"dataset","cifar"),download=True)
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision.transforms as transform
import torchvision.datasets as dsets
import os
import copy
import pandas as pd

from vgg import VGG
import utils
from consts import *


if __name__ == '__main__':
    compose = transform.Compose([transform.Resize((IMAGE_SIZE, IMAGE_SIZE)), transform.ToTensor()])

    training_data = dsets.FashionMNIST(
        root='./data',
        train=True,
        download=FASHION_MNIST,
        transform=compose
    )
    testing_data = dsets.FashionMNIST(
        root='./data',
        train=False,
        download=True,
        transform=compose
    )

    # utils.show_data(testing_data, 3, 'images/test')

    training_loader = Data.DataLoader(dataset=training_data,
                                      batch_size=BATCH_SIZE,
                                      shuffle=True,
                                      num_workers=2)
    testing_loader = Data.DataLoader(dataset=testing_data,
                                     batch_size=500,
                                     shuffle=False,
                                     num_workers=0)

    optimizers, optimizers_m, scripts, scripts_m = [], [], [], []

    torch.manual_seed(0)
    vgg16 = VGG(VGG16_CONFIG)

    optimizers.append(torch.optim.SGD(params=vgg16.parameters(), lr=1e-3, momentum=.9))
    scripts.append('sgd-1e-3_9')

    optimizers.append(torch.optim.SGD(params=vgg16.parameters(), lr=1e-3, momentum=.8))
    scripts.append('sgd-1e-3_8')

    optimizers.append(torch.optim.SGD(params=vgg16.parameters(), lr=1e-3, momentum=.7))
    scripts.append('sgd-1e-3_7')

    loss_function = nn.CrossEntropyLoss()

    if os.path.exists(RECORD_PATH):
        records = pd.read_pickle(RECORD_PATH)
    else:
        records = pd.DataFrame()

    utils.experient(training_loader, testing_loader,
                    vgg16, loss_function, optimizers,
                    scripts, records, 'images/vgg16/momentum')
    # utils.experient(training_loader, testing_loader,
    #                 vgg16_h, loss_function, optimizers_m,
    #                 scripts_m, records, 'images/vgg16_half/momentum')

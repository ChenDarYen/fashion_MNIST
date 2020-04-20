import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision.transforms as transform
import torchvision.datasets as dsets
import pickle

from vgg import VGG
import utils
from consts import *


if __name__ == '__main__':
    torch.manual_seed(0)

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

    training_loader = Data.DataLoader(dataset=training_data, batch_size=BATCH_SIZE, shuffle=True)
    testing_loader = Data.DataLoader(dataset=testing_data, batch_size=1000, shuffle=False)

    vgg16 = VGG(VGG16_CONFIG)
    optimizer = torch.optim.Adam(params=vgg16.parameters(), lr=LR)
    loss_function = nn.CrossEntropyLoss()

    record = []
    cost_list, accuracy_list, time = utils.experiment(training_loader, testing_loader, vgg16, optimizer, loss_function)
    print(cost_list)
    print(accuracy_list)
    print(time)
    record.append({
        'layer': 16,
        'lr': LR,
        'batch_size': BATCH_SIZE,
        'optimizer': {
            'name': 'adam',
            'beta': [.9, .999],
            'epsilon': 1e-8,
            'weight_decay': 0,
        },
        'cost': cost_list,
        'accuracy': accuracy_list,
        'time': time,
    })

    file = open('record.pkl', 'wb')
    pickle.dump(record, file)
    file.close()

    utils.visual_cost_accuracy(cost_list, accuracy_list, 'images/vgg16')
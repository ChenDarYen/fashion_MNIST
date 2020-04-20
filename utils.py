import numpy as np
import torch
import matplotlib.pyplot as plt
import time
from consts import *


def show_sample(sample):
    plt.imshow(sample[0].numpy().squeeze(), cmap='gray')
    plt.title('y = {}'.format(sample[1]))


def show_data(dataset, num, save_path=None):
    for n, sample in enumerate(dataset):
        show_sample(sample)
        if save_path:
            plt.savefig('{}_{}.png'.format(save_path, n))
        plt.show()
        if n == num - 1:
            break


def visual_cost_accuracy(cost_list, accuracy_list, save_path=None, show=True):
    X = np.arange(len(cost_list)) + 1

    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.plot(X, cost_list, color=color)
    ax1.set_xlabel('epoch')
    ax1.set_xticks(X)
    ax1.set_ylabel('Cost', color=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.plot(X, accuracy_list, color=color)
    ax2.set_ylabel('accuracy', color=color)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    else:
        plt.cla()


def calc_time(start, end):
    duration = end - start
    sec = duration % 60
    min = duration // 60
    hour = duration // 3600

    return hour, min, sec


def experiment(training_loader, testing_loader, model, optimizer, loss_function):
    start = time.time()

    test_len = len(testing_loader.dataset)
    batch_size = training_loader.batch_size
    cost_list, accuracy_list = [], []
    for epoch in range(EPOCHS):
        print('{} epoch'.format(epoch))
        cost = 0
        for X, Y in training_loader:
            Z = torch.softmax(model(X), dim=2).view(batch_size, -1)
            loss = loss_function(Z, Y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            cost += loss.item()
            print(loss.item())
        print('cost: {}'.format(cost))

        correct = 0
        for X, Y in testing_loader:
            Z = model(X)
            predict = torch.max(Z, 2)[1].data.squeeze()
            correct += (predict == Y).sum().item()
        print('accuracy: {}'.format(correct / test_len))

        cost_list.append(cost)
        accuracy_list.append(correct / test_len)

    end = time.time()

    return cost_list, accuracy_list, end - start

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from consts import COLORS


def show_loss(loss_lists, labels, title, step=1, save_path=None):
    length = len(loss_lists.iloc[0]) // step
    X = np.linspace(0, 10 * length * step, length)

    for n, (loss_list, label) in enumerate(zip(loss_lists, labels)):
        loss_list = loss_list[::step]
        plt.plot(X, loss_list, color=COLORS[n], label=label)

    plt.legend()
    plt.xlabel('steps')
    plt.ylabel('loss')
    plt.title(title)

    if save_path:
        plt.savefig(save_path)

    plt.show()


if __name__ == '__main__':
    begin = 5
    end = 7

    records = pd.read_pickle('records.pkl')
    loss_lists = records.lost[begin:end]
    labels = records.optimizer[begin:end]

    print(labels)

    show_loss(loss_lists, labels, 'SGD', 3, 'images/sgd_momentum.png')

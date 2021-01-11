import torch
from matplotlib import pyplot as plt


def plot_curve(data, color, value):
    fig = plt.figure()
    plt.plot(range(len(data)), data, color= color)
    plt.legend([value], loc='upper center')
    plt.xlabel('step')
    plt.ylabel(value)
    plt.show()

def plot_image(img, label, name):
    fig = plt.figure()
    for i in range(6):
        plt.subplot(2, 3, i+1)
        plt.tight_layout()
        plt.imshow(img[i][0], cmap='binary', interpolation='none')   #
        plt.title("{}: {}".format(name, label[i].item()))
        plt.xticks([])
        plt.yticks([])
    plt.show()

def one_hot(label, depth=10):
    out = torch.zeros(label.size(0), depth)
    idx = torch.LongTensor(label).view(-1, 1)
    out.scatter_(dim=1, index=idx, value=1)
    return out


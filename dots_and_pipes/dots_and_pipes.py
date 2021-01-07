import matplotlib.pyplot as plt
import numpy as np


def polka_dots(save_fig=False):
    """
    I like the default color pallet here
    :param save_fig:
    :return:
    """
    fig = plt.figure(frameon=True)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    for i in range(100):
        x, y = np.random.randint(0, 20, 2)
        ax.scatter(x, y)

    if save_fig:
        plt.savefig('polka_dots')
    plt.show()


def vert_horz(save_fig=False):
    """
    I like the default color pallet here
    :param save_fig:
    :return:
    """
    fig = plt.figure(frameon=True)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor((0.2, 0.25, 0.23))
    # ax.axis('off')
    for i in range(1000):
        x, y = np.random.randint(0, 100, 2)
        if np.random.weibull(0.9) < x/100.:
            marker = '|'
        else:
            marker = '_'
        color_offset = np.random.random()
        ax.scatter(x, y, marker=marker, c=[(0.7, 0.9, 0.03, color_offset * 0.9)])

    if save_fig:
        plt.savefig('vert_horz')
    plt.show()


def vert_horz_wider(save_fig=False):
    """
    I like the default color pallet here
    :param save_fig:
    :return:
    """
    fig = plt.figure(frameon=True)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor((0.2, 0.25, 0.23))
    # ax.axis('off')
    for i in range(10000):
        x, y = np.random.randint(0, 100, 2)
        if np.random.weibull(0.9) < x/100.:
            marker = '|'
        else:
            marker = '_'
        color_offset = np.random.random()
        ax.scatter(x, y, marker=marker, c=[(0.7, 0.9, 0.03, color_offset * 0.9)], s=100)

    if save_fig:
        plt.savefig('vert_horz_pipez')
    plt.show()

if __name__ == '__main__':
    polka_dots()
    # vert_horz()
    # vert_horz_wider()

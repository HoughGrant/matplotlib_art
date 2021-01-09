import matplotlib.pyplot as plt
import numpy as np


def mscatter(x, y, ax=None, m=None, **kw):
    import matplotlib.markers as mmarkers
    if not ax: ax=plt.gca()
    sc = ax.scatter(x, y, **kw)
    if (m is not None) and (len(m) == len(x)):
        paths = []
        for marker in m:
            if isinstance(marker, mmarkers.MarkerStyle):
                marker_obj = marker
            else:
                marker_obj = mmarkers.MarkerStyle(marker)
            path = marker_obj.get_path().transformed(
                        marker_obj.get_transform())
            paths.append(path)
        sc.set_paths(paths)
    return sc


def polka_dots(save_fig=False, num_points=100):
    """
    I like the default color pallet here
    :param save_fig:
    :return:
    """
    fig = plt.figure(frameon=True)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')

    x_coords = np.random.randint(0, 20, num_points)
    y_coords = np.random.randint(0, 20, num_points)
    colors = [np.random.uniform(0, 1, 3) for x, y in zip(x_coords, y_coords)]

    ax.scatter(x_coords, y_coords, c=colors)
    if save_fig:
        plt.savefig('polka_dots')
    plt.show()


def vert_horz(save_fig=False, num_points: int = 1000):
    """
    I like the default color pallet here
    :param save_fig:
    :return:
    """
    fig = plt.figure(frameon=True)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor((0.2, 0.25, 0.23))
    # ax.axis('off')

    x_coords = np.random.randint(0, 100, num_points)
    y_coords = np.random.randint(0, 100, num_points)
    markers = ['|' if np.random.weibull(0.9) < x/100. else '_' for x in x_coords]
    colors = [(0.7, 0.9, 0.03, np.random.random() * 0.9) for x in x_coords]

    mscatter(x_coords, y_coords, c=colors, m=markers, ax=ax)

    if save_fig:
        plt.savefig('vert_horz')
    plt.show()


def vert_horz_wider(save_fig=False, num_points=10000, marker_size=100):
    """
    I like the default color pallet here
    :param save_fig:
    :return:
    """
    fig = plt.figure(frameon=True)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor((0.2, 0.25, 0.23))
    # ax.axis('off')

    x_coords = np.random.randint(0, 100, num_points)
    y_coords = np.random.randint(0, 100, num_points)
    markers = ['|' if np.random.weibull(0.9) < x/100. else '_' for x in x_coords]
    colors = [(0.7, 0.9, 0.03, np.random.random() * 0.9) for x in x_coords]

    mscatter(x_coords, y_coords, c=colors, m=markers, ax=ax, s=marker_size)

    if save_fig:
        plt.savefig('vert_horz_pipez')
    plt.show()

if __name__ == '__main__':
    # polka_dots()
    # vert_horz(num_points=1000)
    vert_horz_wider()
    # vert_horz_wider(marker_size=1000)

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


def sparse_purple(save_fig=False, num_points=50):
    fig = plt.figure(frameon=True)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor((0.7, 0.2, 0.65))

    x_coords = np.random.randint(0, 50, num_points)
    y_coords = np.random.randint(0, 50, num_points)
    colors = [np.append(np.random.uniform(0, 0.3, 3), 0.1) for x in x_coords]
    markers = np.random.randint(0, 4, num_points)
    mscatter(x_coords, y_coords, c=colors, m=markers, ax=ax, s=1000)

    mask = np.random.randint(0, 2, 40)
    x_coords = 3 + np.arange(0, 40)[::-1]
    y_coords = 4 + np.arange(10, 50)
    colors = (0.7, 0.25, 0.65, 0.45)
    mscatter(x_coords * mask, y_coords * mask, c=colors, m='s', ax=ax, s=200)

    mask = np.random.randint(0, 2, 20)
    x_coords = 3 + np.arange(0, 20)
    y_coords = 4 + np.arange(25, 45)
    colors = (0.7, 0.25, 0.65, 0.25)
    mscatter(x_coords * mask, y_coords * mask, c=colors, m='s', ax=ax, s=800)

    mask = np.random.randint(0, 2, 10)
    x_coords = 3 + np.arange(30, 40)
    y_coords = 4 + np.arange(5, 15)
    colors = (0.7, 0.25, 0.65, 0.25)
    mscatter(x_coords * mask, y_coords * mask, c=colors, m='s', ax=ax, s=800)

    if save_fig:
        plt.savefig('sparse_purple')
    plt.show()


if __name__ == '__main__':
    sparse_purple(num_points=500)

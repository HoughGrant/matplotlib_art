import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from skimage.transform import downscale_local_mean
from sklearn.cluster import KMeans


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


def sparse_gorillaz(save_fig=False, num_points=50, num_shuffles=5, n_clusters=5, filename=None):
    if filename is None:
        filename = 'sparse_gorillaz'
    if n_clusters < 4:
        n_clusters = 4
    url = 'https://i.pinimg.com/originals/a2/75/0c/a2750c2051f6c5eda339bf314d1075e4.jpg'
    image_numpy = io.imread(url)
    image_numpy = downscale_local_mean(image_numpy, (8, 8, 1))

    image_flattened = image_numpy.reshape(-1, image_numpy.shape[-1])
    kmeans = KMeans(n_clusters=n_clusters, random_state=1).fit(image_flattened)
    possible_colors = kmeans.cluster_centers_ / 256.

    for shuffle_index in range(num_shuffles):
        np.random.shuffle(possible_colors)
        face_color = possible_colors[0]
        color1 = possible_colors[1]
        color2 = possible_colors[2]
        color3 = possible_colors[3]

        fig = plt.figure(frameon=True)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_facecolor(face_color)

        x_coords = np.random.randint(0, 50, num_points)
        y_coords = np.random.randint(0, 50, num_points)
        colors = [np.append(color1 + 0 * np.random.uniform(0, 0.05, 3), 0.1) for x in x_coords]
        markers = np.random.randint(0, 4, num_points)
        mscatter(x_coords, y_coords, c=colors, m=markers, ax=ax, s=1000)

        mask = np.random.randint(0, 2, 40)
        x_coords = 3 + np.arange(0, 40)[::-1]
        y_coords = 4 + np.arange(10, 50)
        colors = np.append(color2 + 0 * np.random.uniform(0, 0.05, 3), 0.9)
        # colors = (0.7, 0.25, 0.65, 0.45)
        mscatter(x_coords * mask, y_coords * mask, c=colors, m='s', ax=ax, s=200)

        mask = np.random.randint(0, 2, 20)
        x_coords = 3 + np.arange(0, 20)
        y_coords = 4 + np.arange(25, 45)
        # colors = (0.7, 0.25, 0.65, 0.25)
        colors = np.append(color3 + 0 * np.random.uniform(0, 0.05, 3), 0.1)
        mscatter(x_coords * mask, y_coords * mask, c=colors, m='s', ax=ax, s=800)

        mask = np.random.randint(0, 2, 10)
        x_coords = 3 + np.arange(30, 40)
        y_coords = 4 + np.arange(5, 15)
        colors = np.append(color3 + 0 * np.random.uniform(0, 0.05, 3), 0.1)
        mscatter(x_coords * mask, y_coords * mask, c=colors, m='s', ax=ax, s=800)

        if save_fig:
            plt.savefig(filename + '_' + str(shuffle_index))
    plt.show()

def sparse_pipes_from_image(image_url, save_fig=False, num_points=50, num_shuffles=5, n_clusters=5, filename=None):
    if filename is None:
        filename = 'sparse_gorillaz'
    if n_clusters < 4:
        n_clusters = 4
    image_numpy = io.imread(image_url)
    image_numpy = downscale_local_mean(image_numpy, (8, 8, 1))

    image_flattened = image_numpy.reshape(-1, image_numpy.shape[-1])
    kmeans = KMeans(n_clusters=n_clusters, random_state=1).fit(image_flattened)
    possible_colors = kmeans.cluster_centers_ / 256.

    for shuffle_index in range(num_shuffles):
        np.random.shuffle(possible_colors)
        face_color = possible_colors[0]
        color1 = possible_colors[1]
        color2 = possible_colors[2]
        color3 = possible_colors[3]

        fig = plt.figure(frameon=True)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_facecolor(face_color)

        x_coords = np.random.randint(0, 50, num_points)
        y_coords = np.random.randint(0, 50, num_points)
        colors = [np.append(color1 + 0 * np.random.uniform(0, 0.05, 3), 0.1) for x in x_coords]
        markers = np.random.randint(0, 4, num_points)
        mscatter(x_coords, y_coords, c=colors, m=markers, ax=ax, s=1000)

        mask = np.random.randint(0, 2, 40)
        x_coords = 3 + np.arange(0, 40)[::-1]
        y_coords = 4 + np.arange(10, 50)
        colors = np.append(color2 + 0 * np.random.uniform(0, 0.05, 3), 0.9)
        # colors = (0.7, 0.25, 0.65, 0.45)
        mscatter(x_coords * mask, y_coords * mask, c=colors, m='s', ax=ax, s=200)

        mask = np.random.randint(0, 2, 20)
        x_coords = 3 + np.arange(0, 20)
        y_coords = 4 + np.arange(25, 45)
        # colors = (0.7, 0.25, 0.65, 0.25)
        colors = np.append(color3 + 0 * np.random.uniform(0, 0.05, 3), 0.1)
        mscatter(x_coords * mask, y_coords * mask, c=colors, m='s', ax=ax, s=800)

        mask = np.random.randint(0, 2, 10)
        x_coords = 3 + np.arange(30, 40)
        y_coords = 4 + np.arange(5, 15)
        colors = np.append(color3 + 0 * np.random.uniform(0, 0.05, 3), 0.1)
        mscatter(x_coords * mask, y_coords * mask, c=colors, m='s', ax=ax, s=800)

        if save_fig:
            plt.savefig(filename + '_' + str(shuffle_index))
    plt.show()


if __name__ == '__main__':
    # sparse_purple(num_points=500)
    # sparse_gorillaz(num_points=5000)
    # sparse_gorillaz(num_points=500, filename='sparse_gorillaz_2')
    # sparse_gorillaz(num_points=500, n_clusters=10)
    sparse_pipes_from_image('https://upload.wikimedia.org/wikipedia/commons/thumb/f/f4/The_Scream.jpg/471px-The_Scream.jpg',
                            num_points=5000, filename='scream_pipes')
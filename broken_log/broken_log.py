import matplotlib.pyplot as plt
import numpy as np
import random

from skimage.transform import downscale_local_mean
from sklearn.cluster import KMeans
from skimage import io, color
from typing import Tuple


class ColorPallete:
    def __init__(self, url, n_clusters=8):
        self.url = url
        self.clustered_colors_lab = None
        self.clustered_colors_rgb = None
        self.n_clusters = n_clusters

    def extract_colors(self, remove_white_and_black=True, l_mod=1.0):
        image_rgb = io.imread(self.url)
        image_lab = color.rgb2lab(image_rgb)
        image_lab = downscale_local_mean(image_lab, (8, 8, 1))
        image_flattened = image_lab.reshape(-1, image_lab.shape[-1])
        kmeans_lab = KMeans(n_clusters=self.n_clusters, random_state=1).fit(image_flattened)
        self.clustered_colors_lab = kmeans_lab.cluster_centers_
        if remove_white_and_black:
            self.clustered_colors_lab = self.clustered_colors_lab[(self.clustered_colors_lab[:, 0] > 10) &
                                                                  (self.clustered_colors_lab[:, 0] < 95)]
        self.change_l(l_mod)
        self.clustered_colors_rgb = color.lab2rgb(self.clustered_colors_lab)
        return self.clustered_colors_rgb

    def plot_rgb_colors(self, rgb_colors):
        all_colors = np.ones((rgb_colors.shape[0] * 10, 5, 3))
        for i, rgb in enumerate(rgb_colors):
            all_colors[i * 10:(i + 1) * 10] = rgb
        plt.figure()
        plt.imshow(all_colors)
        plt.show()

    def plot_available_colors(self):
        if self.clustered_colors_rgb is None:
            self.extract_colors()
        self.plot_rgb_colors(self.clustered_colors_rgb)

    def change_l(self, mod_coeff):
        modified_lab = self.clustered_colors_lab[:]
        modified_lab[:, 0] *= mod_coeff
        modified_colors_rgb = color.lab2rgb(modified_lab)
        self.clustered_colors_rgb = modified_colors_rgb

    def get_least_light_color(self):
        if self.clustered_colors_rgb is None:
            self.extract_colors()
        lightest_lab = self.clustered_colors_lab[np.argmin(self.clustered_colors_lab[0, :])]
        return color.lab2rgb(lightest_lab)

    def get_most_light_color(self):
        if self.clustered_colors_rgb is None:
            self.extract_colors()
        lightest_lab = self.clustered_colors_lab[np.argmax(self.clustered_colors_lab[0, :])]
        return color.lab2rgb(lightest_lab)


class CreateBrokenGraph:
    def __init__(self, color_pallete: ColorPallete, background_color: Tuple[float] = None):
        self.color_pallete = color_pallete
        if background_color is None:
            self.background_color = color_pallete.get_most_light_color()
            # self.background_color = color_pallete.get_least_light_color()
        else:
            self.background_color = background_color
        self.num_splits = color_pallete.n_clusters
        self.pts = None
        pass

    def generate_data(self, sample_length=int(1e4)):
        pts = np.cumsum(np.random.uniform(-1000, 1005, sample_length))
        perturb = np.random.exponential(1000, sample_length)
        pts += perturb
        pts += np.abs(np.min(pts))
        if self.pts is None:
            self.pts = pts
        return pts

    def plot_figure(self, n_lines=3, **kwargs):
        rgb_colors = self.color_pallete.extract_colors(**kwargs)

        plt.rcParams["figure.facecolor"] = self.background_color
        fig, axes = plt.subplots(self.num_splits, 1, sharex='all', frameon=True)
        fig.tight_layout()
        fig.subplots_adjust(hspace=-0.05)  # adjust space between axes

        for _ in range(n_lines):
            alpha = 1.0
            for index, ax in enumerate(axes):
                pts = self.generate_data()
                color = np.append(rgb_colors[index], alpha)
                linewidth = np.random.exponential(2)
                ax.semilogy(pts, color=color, alpha=alpha, linewidth=linewidth)
                alpha *= 0.9

        max = 2.0 * np.max(self.pts)
        min = np.min(self.pts)
        diff = (max - min) / self.num_splits
        min -= diff * 0.1
        for ax in axes[:-1]:
            ax.set_ylim(max - diff, max)
            max *= np.random.uniform(0.5, 1.1)
        axes[-1].set_ylim(min, max - diff)

        axes[0].spines['bottom'].set_visible(False)
        for ax in axes[1:-2]:
            ax.spines['bottom'].set_visible(False)
            ax.spines['top'].set_visible(False)

        axes[-1].spines['top'].set_visible(False)
        axes[0].xaxis.tick_top()
        axes[0].tick_params(labeltop=False)
        axes[-1].xaxis.tick_bottom()

        for ax in axes:
            ax.axis('off')

        plt.show()


def run_1():
    url = 'https://i.pinimg.com/originals/a2/75/0c/a2750c2051f6c5eda339bf314d1075e4.jpg'
    # url = 'https://upload.wikimedia.org/wikipedia/commons/6/6d/Blue-and-Yellow-Macaw.jpg'
    color_pallete = ColorPallete(url, n_clusters=2)
    broken_graph = CreateBrokenGraph(color_pallete)
    for _ in range(5):
        broken_graph.plot_figure(n_lines=10, remove_white_and_black=False)


if __name__ == '__main__':
    run_1()

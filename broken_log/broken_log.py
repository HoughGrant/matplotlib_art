import matplotlib.pyplot as plt
import numpy as np

from skimage.transform import downscale_local_mean
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
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
        # gaussian_mixture_lab = GaussianMixture(n_components=self.n_clusters, random_state=1).fit(image_flattened)
        # self.clustered_colors_lab = gaussian_mixture_lab.means_
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
        lower_step = np.random.uniform(-1050, -950)
        upper_step = np.random.uniform(1000, 1050)

        pts = np.cumsum(np.random.uniform(lower_step, upper_step, sample_length))
        perturb = np.random.exponential(1000, sample_length)
        pts += perturb
        pts += np.abs(np.min(pts))
        if self.pts is None:
            self.pts = pts
        return pts

    def plot_figure(self, n_lines=3, **kwargs):
        rgb_colors = self.color_pallete.extract_colors(**kwargs)
        background_color = self.background_color * np.random.uniform(0.9, 1.1, 3)
        background_color[background_color > 1.0] = 1.0
        plt.rcParams["figure.facecolor"] = background_color

        fig, axes = plt.subplots(self.num_splits, 1, sharex='all', frameon=True)
        fig.tight_layout()
        fig.subplots_adjust(hspace=-0.05)  # adjust space between axes

        for _ in range(n_lines):
            alpha_mod = 1.0
            for index, ax in enumerate(axes):
                pts = self.generate_data()
                linewidth = np.min([np.random.exponential(2), 5])
                alpha = linewidth / 5 * alpha_mod
                rgb_random_factors = np.random.uniform(0.9, 1.1, 3)
                plot_colors = rgb_colors[index] * rgb_random_factors
                plot_colors[plot_colors > 1.0] = 1.0
                color = np.append(plot_colors, alpha)
                ax.semilogy(pts, color=color, alpha=alpha, linewidth=linewidth)
                alpha_mod *= 0.9

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
    # url = 'https://i.pinimg.com/originals/a2/75/0c/a2750c2051f6c5eda339bf314d1075e4.jpg'
    url = 'https://upload.wikimedia.org/wikipedia/commons/6/6d/Blue-and-Yellow-Macaw.jpg'
    color_pallete = ColorPallete(url, n_clusters=2)
    broken_graph = CreateBrokenGraph(color_pallete)
    for i in range(5):
        broken_graph.plot_figure(n_lines=50, remove_white_and_black=False)
        # plt.savefig('ice_table_variation_' + str(i))
        # plt.close()


def run_2():
    # url = 'https://upload.wikimedia.org/wikipedia/commons/6/6d/Blue-and-Yellow-Macaw.jpg'
    url = 'https://i.pinimg.com/originals/a2/75/0c/a2750c2051f6c5eda339bf314d1075e4.jpg'
    color_pallete = ColorPallete(url, n_clusters=10)
    broken_graph = CreateBrokenGraph(color_pallete)

    def generate_data_new(self, sample_length=int(1e4)):
        pts = np.sin(np.arange(sample_length))
        perturb = np.random.exponential(1000, sample_length)
        pts += perturb
        pts += np.abs(np.min(pts))
        if self.pts is None:
            self.pts = pts
        return pts

    import types

    broken_graph.generate_data = types.MethodType(generate_data_new, broken_graph)
    broken_graph.plot_figure(n_lines=5, remove_white_and_black=False)


def waterfeature():
    def generate_data_new(self, sample_length=int(1e4)):
        x_pts_start = np.random.uniform(0., 0.)
        x_pts_end = np.random.uniform(np.pi, np.pi)
        x_pts = np.linspace(x_pts_start, x_pts_end, sample_length)  # / (2 * np.pi * sample_length)
        sin_freq = np.random.randint(1, 3)
        # sin_amp_probs = [0.5, 0.1, 0.1, 0.2, 0.3]
        sin_amp = np.random.choice([100, 200, 300, 400, 500],
                                   # p=sin_amp_probs
                                   )
        y_pts = sin_amp * np.sin(x_pts * (2 * sin_freq - 1) * np.pi)
        perturb = np.random.exponential(10, sample_length) * (-1) ** np.random.randint(0, 2)
        y_pts += perturb
        y_pts += np.abs(np.min(y_pts))
        if self.y_pts is None:
            self.y_pts = y_pts
        if self.x_pts is None:
            self.x_pts = x_pts
        return x_pts, y_pts

    def plot_figure_new(self, n_lines=3, **kwargs):
        rgb_colors = self.color_pallete.extract_colors(**kwargs)
        background_color = self.background_color * np.random.uniform(0.9, 1.1, 3)
        background_color[background_color > 1.0] = 1.0
        plt.rcParams["figure.facecolor"] = background_color

        fig, axes = plt.subplots(self.num_splits, 1, sharex='all', frameon=True)
        fig.tight_layout()
        fig.subplots_adjust(hspace=-0.05)  # adjust space between axes

        for _ in range(n_lines):
            top_linewidth = 1.5
            for index, ax in enumerate(axes):
                alpha_mod = 0.9
                x_pts, y_pts = self.generate_data()
                linewidth = np.min([np.random.exponential(0.5), top_linewidth])
                alpha = linewidth / top_linewidth * alpha_mod
                rgb_random_factors = np.random.uniform(0.85, 1.15, 3)
                plot_colors = rgb_colors[index] * rgb_random_factors
                plot_colors[plot_colors > 1.0] = 1.0
                color = np.append(plot_colors, alpha)
                ax.semilogy(x_pts, y_pts, color=color, alpha=alpha, linewidth=linewidth)
                alpha_mod *= 0.9

        max = 1.0 * np.max(self.y_pts)
        min = np.min(self.y_pts)
        diff = (max - min) / self.num_splits
        min *= 0.9
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

    import types

    # url = 'https://upload.wikimedia.org/wikipedia/commons/6/6d/Blue-and-Yellow-Macaw.jpg'
    url = 'https://i.pinimg.com/originals/a2/75/0c/a2750c2051f6c5eda339bf314d1075e4.jpg'
    color_pallete = ColorPallete(url, n_clusters=5)
    broken_graph = CreateBrokenGraph(color_pallete)
    broken_graph.y_pts = None
    broken_graph.x_pts = None
    broken_graph.generate_data = types.MethodType(generate_data_new, broken_graph)
    broken_graph.plot_figure = types.MethodType(plot_figure_new, broken_graph)
    broken_graph.plot_figure(n_lines=50, remove_white_and_black=False)

def frogsymmetric():
    def generate_data_new(self, sample_length=int(1e4)):
        x_pts_start = np.random.uniform(0., 0.)
        x_pts_end = np.random.uniform(np.pi, np.pi)
        x_pts = np.linspace(x_pts_start, x_pts_end, sample_length)  # / (2 * np.pi * sample_length)
        sin_freq = np.random.randint(1, 3)
        sin_amp_probs = [0.05, 0.05, 0.1, 0.5, 0.3]
        sin_amp = np.random.choice([100, 200, 300, 400, 500],
                                   p=sin_amp_probs
                                   )
        y_pts = sin_amp * np.sin(x_pts * (2 * sin_freq - 1) * np.pi)
        perturb = np.random.exponential(10, sample_length) * (-1) ** np.random.randint(0, 2)
        y_pts += perturb
        y_pts += np.abs(np.min(y_pts))
        if self.y_pts is None:
            self.y_pts = y_pts
        if self.x_pts is None:
            self.x_pts = x_pts
        return x_pts, y_pts

    def plot_figure_new(self, n_lines=3, **kwargs):
        rgb_colors = self.color_pallete.extract_colors(**kwargs)
        background_color = self.background_color * np.random.uniform(0.9, 1.1, 3)
        background_color[background_color > 1.0] = 1.0
        plt.rcParams["figure.facecolor"] = background_color

        fig, axes = plt.subplots(self.num_splits, 1, sharex='all', frameon=True)
        fig.tight_layout()
        fig.subplots_adjust(hspace=-0.05)  # adjust space between axes

        for _ in range(n_lines):
            top_linewidth = 1.5
            for index, ax in enumerate(axes):
                alpha_mod = 0.99
                x_pts, y_pts = self.generate_data()
                linewidth = np.min([np.random.exponential(0.5), top_linewidth])
                alpha = linewidth / top_linewidth * alpha_mod
                rgb_random_factors = np.random.uniform(0.85, 1.15, 3)
                plot_colors = rgb_colors[index] * rgb_random_factors
                plot_colors[plot_colors > 1.0] = 1.0
                color = np.append(plot_colors, alpha)
                ax.semilogy(x_pts, y_pts, color=color, alpha=alpha, linewidth=linewidth)
                alpha_mod *= 0.95

        max = 1.0 * np.max(self.y_pts)
        min = np.min(self.y_pts)
        diff = (max - min) / self.num_splits
        min *= 0.9
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

    import types

    url = 'https://upload.wikimedia.org/wikipedia/commons/6/6d/Blue-and-Yellow-Macaw.jpg'
    # url = 'https://i.pinimg.com/originals/a2/75/0c/a2750c2051f6c5eda339bf314d1075e4.jpg'
    color_pallete = ColorPallete(url, n_clusters=4)
    broken_graph = CreateBrokenGraph(color_pallete)
    broken_graph.y_pts = None
    broken_graph.x_pts = None
    broken_graph.generate_data = types.MethodType(generate_data_new, broken_graph)
    broken_graph.plot_figure = types.MethodType(plot_figure_new, broken_graph)
    broken_graph.plot_figure(n_lines=5, remove_white_and_black=False)


def faces_in_clouds():
    def generate_data_new(self, sample_length=int(1e4)):
        x_pts_start = np.random.uniform(0., 0.)
        x_pts_end = np.random.uniform(np.pi, np.pi)
        x_pts = np.linspace(x_pts_start, x_pts_end, sample_length)  # / (2 * np.pi * sample_length)

        sin_freq_probs = [0.4, 0.2, 0.1, 0.01, 0.01, 0.001]
        sin_freq_probs /= np.sum(sin_freq_probs)
        sin_freq = np.random.choice([1, 2, 3, 4, 5, 6], p=sin_freq_probs)

        sin_amp_probs = [0.05, 0.05, 0.1, 0.5, 0.3]
        sin_amp = np.random.choice([100, 200, 300, 400, 500], p=sin_amp_probs)
        y_pts = sin_amp * np.sin(x_pts * (2 * sin_freq - 1) * np.pi)
        perturb = np.random.exponential(10, sample_length) * (-1) ** np.random.randint(0, 2)
        y_pts += perturb
        y_pts += np.abs(np.min(y_pts))
        if self.y_pts is None:
            self.y_pts = y_pts
        if self.x_pts is None:
            self.x_pts = x_pts
        return x_pts, y_pts

    def plot_figure_new(self, n_lines=3, **kwargs):
        rgb_colors = self.color_pallete.extract_colors(**kwargs)
        background_color = self.background_color * np.random.uniform(0.9, 1.1, 3)
        background_color[background_color > 1.0] = 1.0
        plt.rcParams["figure.facecolor"] = background_color

        fig, axes = plt.subplots(self.num_splits, 1, sharex='all', frameon=True)
        fig.tight_layout()
        fig.subplots_adjust(hspace=-0.05)  # adjust space between axes

        for _ in range(n_lines):
            top_linewidth = 1.5
            for index, ax in enumerate(axes):
                alpha_mod = 0.99
                x_pts, y_pts = self.generate_data()
                linewidth = np.min([np.random.exponential(0.5), top_linewidth])
                alpha = linewidth / top_linewidth * alpha_mod
                rgb_random_factors = np.random.uniform(0.85, 1.15, 3)
                plot_colors = rgb_colors[index] * rgb_random_factors
                plot_colors[plot_colors > 1.0] = 1.0
                color = np.append(plot_colors, alpha)
                ax.semilogy(x_pts, y_pts, color=color, alpha=alpha, linewidth=linewidth)
                alpha_mod *= 0.95

        max = 1.0 * np.max(self.y_pts)
        min = np.min(self.y_pts)
        diff = (max - min) / self.num_splits
        min *= 0.9
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

    import types

    # url = 'https://upload.wikimedia.org/wikipedia/commons/6/6d/Blue-and-Yellow-Macaw.jpg'
    url = 'https://i.pinimg.com/originals/a2/75/0c/a2750c2051f6c5eda339bf314d1075e4.jpg'
    color_pallete = ColorPallete(url, n_clusters=4)
    broken_graph = CreateBrokenGraph(color_pallete)
    broken_graph.y_pts = None
    broken_graph.x_pts = None
    broken_graph.generate_data = types.MethodType(generate_data_new, broken_graph)
    broken_graph.plot_figure = types.MethodType(plot_figure_new, broken_graph)
    broken_graph.plot_figure(n_lines=5, remove_white_and_black=False)

    # sample_length = 10000
    # for _ in range(10):
    #     x_pts_start = np.random.uniform(0., 0.)
    #     x_pts_end = np.random.uniform(np.pi, np.pi)
    #     x_pts = np.linspace(x_pts_start, x_pts_end, sample_length)  # / (2 * np.pi * sample_length)
    #     sin_freq = np.random.randint(1, 10)
    #     y_pts = 10000 * np.sin(x_pts * (sin_freq * np.pi))
    #     perturb = np.random.exponential(100, sample_length) * (-1) ** np.random.randint(0, 2)
    #     y_pts += perturb
    #     y_pts += np.abs(np.min(y_pts))
    #     plt.plot(x_pts, y_pts)
    #     plt.show()


def sin_on_sin():
    def generate_data_new(self, sample_length=int(1e4)):
        x_pts_start = np.random.uniform(0., 0.)
        x_pts_end = np.random.uniform(np.pi, np.pi)
        x_pts = np.linspace(x_pts_start, x_pts_end, sample_length)  # / (2 * np.pi * sample_length)

        sin_freq_probs = [0.4, 0.2, 0.01, 0.001]
        sin_freq_probs /= np.sum(sin_freq_probs)
        sin_freq = np.random.choice([1, 2, 3, 4], p=sin_freq_probs)

        sin_amp_probs = [0.05, 0.05, 0.1, 0.5, 0.3]
        sin_amp = np.random.choice([1000, 2000, 3000, 4000, 5000], p=sin_amp_probs)
        y_pts = sin_amp * np.sin(x_pts * (2 * sin_freq - 1) * np.pi)
        y_pts_higher_freq = np.random.uniform(0.1, 0.4) * sin_amp * np.sin(
            x_pts * (np.random.uniform(2, 5) * 2 * sin_freq - 1) * np.pi)
        perturb = np.random.exponential(100, sample_length) * (-1) ** np.random.randint(0, 2)
        y_pts += perturb
        y_pts += y_pts_higher_freq
        y_pts += np.abs(np.min(y_pts))
        if self.y_pts is None:
            self.y_pts = y_pts
        if self.x_pts is None:
            self.x_pts = x_pts
        return x_pts, y_pts

    def plot_figure_new(self, n_lines=3, **kwargs):
        rgb_colors = self.color_pallete.extract_colors(**kwargs)
        background_color = self.background_color * np.random.uniform(0.9, 1.1, 3)
        background_color[background_color > 1.0] = 1.0
        plt.rcParams["figure.facecolor"] = background_color

        fig, axes = plt.subplots(self.num_splits, 1, sharex='all', frameon=True)
        fig.tight_layout()
        fig.subplots_adjust(hspace=-0.05)  # adjust space between axes

        for _ in range(n_lines):
            top_linewidth = 1.5
            for index, ax in enumerate(axes):
                alpha_mod = 0.99
                x_pts, y_pts = self.generate_data()
                linewidth = np.min([np.random.exponential(0.5), top_linewidth])
                alpha = linewidth / top_linewidth * alpha_mod
                rgb_random_factors = np.random.uniform(0.85, 1.15, 3)
                plot_colors = rgb_colors[index] * rgb_random_factors
                plot_colors[plot_colors > 1.0] = 1.0
                color = np.append(plot_colors, alpha)
                ax.semilogy(x_pts, y_pts, color=color, alpha=alpha, linewidth=linewidth)
                alpha_mod *= 0.8

        max = 1.0 * np.max(self.y_pts)
        min = np.min(self.y_pts)
        diff = (max - min) / self.num_splits
        min *= 0.9
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

    import types

    # url = 'https://upload.wikimedia.org/wikipedia/commons/6/6d/Blue-and-Yellow-Macaw.jpg'
    url = 'https://i.pinimg.com/originals/a2/75/0c/a2750c2051f6c5eda339bf314d1075e4.jpg'
    color_pallete = ColorPallete(url, n_clusters=2)
    broken_graph = CreateBrokenGraph(color_pallete)
    broken_graph.y_pts = None
    broken_graph.x_pts = None
    broken_graph.generate_data = types.MethodType(generate_data_new, broken_graph)
    broken_graph.plot_figure = types.MethodType(plot_figure_new, broken_graph)
    broken_graph.plot_figure(n_lines=10, remove_white_and_black=False)


def spooky():
    def find_nearest_idx(array, value):
        array = np.asarray(array)
        dists = np.linalg.norm(array - value, axis=1)
        idx = (np.abs(dists)).argmin()
        return idx

    def generate_data_new(self, sample_length=int(1e4)):
        x_pts_start = np.random.uniform(0., 0.)
        x_pts_end = np.random.uniform(np.pi, np.pi)
        x_pts = np.linspace(x_pts_start, x_pts_end, sample_length)  # / (2 * np.pi * sample_length)

        sin_freq_probs = [0.4, 0.2, 0.01, 0.001]
        sin_freq_probs /= np.sum(sin_freq_probs)
        sin_freq = np.random.choice([1, 2, 3, 4], p=sin_freq_probs)

        sin_amp_probs = [0.05, 0.05, 0.1, 0.5, 0.3]
        sin_amp = np.random.choice([1000, 2000, 3000, 4000, 5000], p=sin_amp_probs)
        y_pts = np.random.uniform(0.95, 1.05) * sin_amp * np.sin(x_pts * (2 * sin_freq - 1) * np.pi)
        y_pts_higher_freq = np.random.uniform(0.05, 0.2) * sin_amp * np.sin(
            x_pts * (np.random.uniform(2, 5) * 2 * sin_freq - 1) * np.pi)
        perturb = np.random.exponential(100, sample_length) * (-1) ** np.random.randint(0, 2)
        y_pts += perturb
        y_pts += y_pts_higher_freq
        y_pts += np.abs(np.min(y_pts))
        if self.y_pts is None:
            self.y_pts = y_pts
        if self.x_pts is None:
            self.x_pts = x_pts
        return x_pts, y_pts

    def plot_figure_new(self, n_lines=3, **kwargs):
        rgb_colors = self.color_pallete.extract_colors(**kwargs)
        background_color = self.background_color * np.random.uniform(0.8, 1.2, 3)

        closest_idx = find_nearest_idx(rgb_colors, self.background_color)
        rgb_colors = np.delete(rgb_colors, closest_idx, axis=0)

        background_color[background_color > 1.0] = 1.0
        plt.rcParams["figure.facecolor"] = background_color

        fig, axes = plt.subplots(self.num_splits - 1, 1, sharex='all', frameon=True)
        fig.tight_layout()
        fig.subplots_adjust(hspace=-0.05)  # adjust space between axes

        for _ in range(n_lines):
            top_linewidth = 1.5
            alpha_mod = 0.9
            for index, ax in enumerate(axes):
                x_pts, y_pts = self.generate_data()
                linewidth = np.min([np.random.exponential(0.5), top_linewidth])
                alpha = linewidth / top_linewidth * alpha_mod
                rgb_random_factors = np.random.uniform(0.85, 1.15, 3)
                from skimage import color
                # tmp_color_lab = color.rgb2lab(rgb_colors[index])
                # if tmp_color_lab[0] < 40:
                #     tmp_color_lab[0] *= 10
                # tmp_color_rgb = color.lab2rgb(tmp_color_lab)

                tmp_color_hsv = color.rgb2hsv(rgb_colors[index])
                # print(tmp_color_hsv)
                if tmp_color_hsv[1] < 0.20:
                    tmp_color_hsv[1] *= 2.0
                    tmp_color_hsv[1] = np.min([tmp_color_hsv[1], 0.8])
                tmp_color_rgb = color.hsv2rgb(tmp_color_hsv)

                plot_colors = tmp_color_rgb * rgb_random_factors
                plot_colors[plot_colors > 1.0] = 1.0
                line_color = np.append(plot_colors, alpha)
                ax.semilogy(x_pts, y_pts, color=line_color, alpha=alpha, linewidth=linewidth)
                # alpha_mod *= 0.8

        max = 1.0 * np.max(self.y_pts)
        min = np.min(self.y_pts)
        diff = (max - min) / self.num_splits
        min *= 0.9
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

    import types

    # url = 'https://upload.wikimedia.org/wikipedia/commons/6/6d/Blue-and-Yellow-Macaw.jpg'
    url = 'https://i.pinimg.com/originals/a2/75/0c/a2750c2051f6c5eda339bf314d1075e4.jpg'
    color_pallete = ColorPallete(url, n_clusters=10)
    broken_graph = CreateBrokenGraph(color_pallete)
    broken_graph.y_pts = None
    broken_graph.x_pts = None
    broken_graph.generate_data = types.MethodType(generate_data_new, broken_graph)
    broken_graph.plot_figure = types.MethodType(plot_figure_new, broken_graph)
    broken_graph.plot_figure(n_lines=20, remove_white_and_black=False)


def the_fall():
    def find_nearest_idx(array, value):
        array = np.asarray(array)
        dists = np.linalg.norm(array - value, axis=1)
        idx = (np.abs(dists)).argmin()
        return idx

    def generate_data_new(self, sample_length=int(1e4)):
        x_pts_start = np.random.uniform(0., 0.)
        x_pts_end = np.random.uniform(np.pi, np.pi)
        x_pts = np.linspace(x_pts_start, x_pts_end, sample_length)  # / (2 * np.pi * sample_length)

        sin_freq_probs = [0.4, 0.2, 0.01, 0.001]
        sin_freq_probs /= np.sum(sin_freq_probs)
        sin_freq = np.random.choice([1, 2, 3, 4], p=sin_freq_probs)

        sin_amp_probs = [0.05, 0.05, 0.1, 0.5, 0.3]
        sin_amp = np.random.choice([1000, 2000, 3000, 4000, 5000], p=sin_amp_probs)
        y_pts = np.random.uniform(0.95, 1.05) * sin_amp * np.sin(x_pts * (2 * sin_freq - 1) * np.pi)
        y_pts_higher_freq = np.random.uniform(0.05, 0.2) * sin_amp * np.sin(
            x_pts * (np.random.uniform(2, 5) * 2 * sin_freq - 1) * np.pi)
        perturb = np.random.exponential(100, sample_length) * (-1) ** np.random.randint(0, 2)
        y_pts += perturb
        y_pts += y_pts_higher_freq
        y_pts += np.abs(np.min(y_pts))
        if self.y_pts is None:
            self.y_pts = y_pts
        if self.x_pts is None:
            self.x_pts = x_pts
        return x_pts, y_pts

    def plot_figure_new(self, n_lines=3, **kwargs):
        rgb_colors = self.color_pallete.extract_colors(**kwargs)
        background_color = self.background_color * np.random.uniform(0.8, 1.2, 3)

        closest_idx = find_nearest_idx(rgb_colors, self.background_color)
        rgb_colors = np.delete(rgb_colors, closest_idx, axis=0)

        background_color[background_color > 1.0] = 1.0
        plt.rcParams["figure.facecolor"] = background_color

        fig, axes = plt.subplots(self.num_splits - 1, 1, sharex='all', frameon=True)
        fig.tight_layout()
        fig.subplots_adjust(hspace=-0.05)  # adjust space between axes

        for _ in range(n_lines):
            top_linewidth = 1.5
            alpha_mod = 0.9
            for index, ax in enumerate(axes):
                x_pts, y_pts = self.generate_data()
                linewidth = np.min([np.random.exponential(0.5), top_linewidth])
                alpha = linewidth / top_linewidth * alpha_mod
                rgb_random_factors = np.random.uniform(0.85, 1.15, 3)
                from skimage import color
                # tmp_color_lab = color.rgb2lab(rgb_colors[index])
                # if tmp_color_lab[0] < 40:
                #     tmp_color_lab[0] *= 10
                # tmp_color_rgb = color.lab2rgb(tmp_color_lab)

                tmp_color_hsv = color.rgb2hsv(rgb_colors[index])
                # print(tmp_color_hsv)
                if tmp_color_hsv[1] < 0.20:
                    tmp_color_hsv[1] *= 2.0
                    tmp_color_hsv[1] = np.min([tmp_color_hsv[1], 0.8])
                tmp_color_rgb = color.hsv2rgb(tmp_color_hsv)

                plot_colors = tmp_color_rgb * rgb_random_factors
                plot_colors[plot_colors > 1.0] = 1.0
                line_color = np.append(plot_colors, alpha)
                ax.semilogy(x_pts, y_pts, color=line_color, alpha=alpha, linewidth=linewidth)
                # alpha_mod *= 0.8

        max = 1.0 * np.max(self.y_pts)
        min = np.min(self.y_pts)
        diff = (max - min) / self.num_splits
        min *= 0.9
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

    import types

    # url = 'https://upload.wikimedia.org/wikipedia/commons/6/6d/Blue-and-Yellow-Macaw.jpg'
    # url = 'https://i.pinimg.com/originals/a2/75/0c/a2750c2051f6c5eda339bf314d1075e4.jpg'
    # url = 'https://upload.wikimedia.org/wikipedia/en/d/d1/Plasticbeach452.jpg'
    url = 'https://upload.wikimedia.org/wikipedia/en/a/a2/The_Fall_%28Gorillaz_album%29_cover.jpg'
    color_pallete = ColorPallete(url, n_clusters=5)
    broken_graph = CreateBrokenGraph(color_pallete)
    broken_graph.y_pts = None
    broken_graph.x_pts = None
    broken_graph.generate_data = types.MethodType(generate_data_new, broken_graph)
    broken_graph.plot_figure = types.MethodType(plot_figure_new, broken_graph)
    for _ in range(5):
        broken_graph.plot_figure(n_lines=20, remove_white_and_black=False)


if __name__ == '__main__':
    # run_1()
    # run_2()
    # waterfeature()
    # frogsymmetric()
    # faces_in_clouds()
    # sin_on_sin()
    spooky()

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.markers import MarkerStyle
from skimage import io, color
import randomcolor
from matplotlib.gridspec import GridSpec


def run_1():
    fig, axes = plt.subplots(3, 3, subplot_kw={'projection': 'polar'})
    n_points = 100

    for sub_ax in axes:
        for ax in sub_ax:
            r = 2 * np.random.rand(n_points)
            theta = np.random.uniform(0, 360) + np.arange(n_points) / 50
            area = 20 * r ** 2
            colors = theta
            ax.set_facecolor((0.6, 0.65, 0.67))
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            ax.set_thetagrids([])
            ax.set_rgrids([])
            ax.scatter(theta, r, c=colors, s=area, cmap='hsv', alpha=0.75)
    plt.show()


def run_2():
    fig = plt.figure(constrained_layout=False, frameon=False)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor((0.37, 0.37, 0.35))

    n_rows = 2
    n_flowers = 5
    gs = GridSpec(n_rows, n_flowers, figure=fig, wspace=0, hspace=0.05)
    axes = []
    for i in range(n_rows):
        for j in range(n_flowers):
            axes.append(fig.add_subplot(gs[i, j], projection='polar'))

    n_points = 100
    rand_color = randomcolor.RandomColor()
    hue = "yellow"

    for ax in axes:

        stem_color = np.array(rand_color.generate(hue="green",
                                                  luminosity="dark",
                                                  count=1,
                                                  format_='Array_rgb')) / 256.

        flower_background_color = np.array(
            rand_color.generate(hue="yellow", luminosity="dark", count=1, format_='Array_rgb')) / 256.
        flower_background_color = flower_background_color[0]

        all_colors_rgb = np.array(rand_color.generate(hue=hue,
                                                      count=n_points,
                                                      format_='Array_rgb')) / 256.
        all_colors_hsv = np.array([color.rgb2hsv(tmp_color) for tmp_color in all_colors_rgb])
        all_colors_hsv[:, 2] *= np.linspace(0.3, 0.9, n_points)

        all_colors_rgb = color.hsv2rgb(all_colors_hsv)

        r = 1.5 * np.arange(n_points)
        radial_freq = np.random.randint(1, 3)
        theta = np.random.uniform(0, 360) + np.arange(n_points) * radial_freq
        area = 0.08 * r ** 1.8

        ax.set_facecolor(flower_background_color)
        ax.set_thetagrids([270], labels='l', fontsize=40, color=stem_color[0])
        ax.set_rgrids([])
        ax.set_frame_on(False)
        marker = MarkerStyle(marker='o',
                             # fillstyle='bottom'
                             )
        ax.scatter(theta, r, c=all_colors_rgb, s=area, cmap='hsv', alpha=0.95, marker=marker)
    plt.tight_layout()
    plt.show()


def flower_field():
    fig = plt.figure(constrained_layout=False, frameon=False)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor((0.37, 0.37, 0.35))

    n_flowers = 150
    n_points = 100

    flower_size_min = 0.075
    flower_size_max = 0.15

    rand_color = randomcolor.RandomColor()
    axes = []

    exponent = 2.5
    for flower_index in range(n_flowers):
        size = np.random.uniform(flower_size_min, flower_size_max)
        axes.append(fig.add_axes((np.random.uniform(-0.1, 0.9),
                                  np.random.uniform(-0.1, 0.9),
                                  size,
                                  size),
                                 projection='polar'))

        hue = np.random.choice(['blue', 'yellow', 'purple', 'red'])
        flower_background_color = np.array(
            rand_color.generate(hue="yellow", luminosity="dark", count=1, format_='Array_rgb')) / 256.
        flower_background_color = flower_background_color[0]

        all_colors_rgb = np.array(rand_color.generate(hue=hue,
                                                      luminosity='bright',
                                                      count=n_points,
                                                      format_='Array_rgb')) / 256.
        all_colors_hsv = np.array([color.rgb2hsv(tmp_color) for tmp_color in all_colors_rgb])
        all_colors_hsv[:, 2] *= np.linspace(0.1, 1.0, n_points)
        all_colors_rgb = color.hsv2rgb(all_colors_hsv)

        r = 1.5 * np.arange(n_points)
        radial_freq = np.random.randint(3, 6)
        theta = np.random.uniform(0, 360) + np.arange(n_points) * radial_freq

        # exponent = np.random.uniform(1.5, 2.2)
        exponent *= 0.95

        area = 0.008 * r ** exponent * (size / flower_size_max)

        axes[-1].set_facecolor(flower_background_color)
        axes[-1].set_thetagrids([])
        # axes[-1].set_thetagrids([270], labels='l', fontsize=40, color=stem_color[0])
        axes[-1].set_rgrids([])
        axes[-1].set_frame_on(False)
        fillstyle = np.random.choice(['full', 'left', 'right', 'bottom', 'top', 'none'])
        marker = MarkerStyle(marker='o',
                             fillstyle=fillstyle
                             )
        axes[-1].scatter(theta, r, c=all_colors_rgb, s=area, cmap='hsv', alpha=0.95, marker=marker)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    flower_field()

import matplotlib.pyplot as plt
import numpy as np
from skimage import color
import randomcolor


def run_1(save_fig=False):
    """
    I like the default color pallet here
    :param save_fig:
    :return:
    """
    n_lines = 10
    n_points = 1000
    hue = np.random.choice(['blue', 'purple', 'red'])
    luminosity = None
    rand_color = randomcolor.RandomColor()
    background_color = np.array(rand_color.generate(hue=hue,
                                                    luminosity=luminosity,
                                                    format_='Array_rgb')) / 256.
    plt.rcParams["figure.facecolor"] = background_color[0]
    fig, axes = plt.subplots(n_lines, sharex=True, frameon=True)
    offset = 1
    rand_color = randomcolor.RandomColor()
    value = 1.0

    for ax in axes:
        n_points /= 2.
        color_rgb = np.array(rand_color.generate(hue=hue,
                                                 luminosity=luminosity,
                                                 format_='Array_rgb')) / 256.
        color_rgb = color_rgb[0]
        color_hsv = color.rgb2hsv(color_rgb)
        color_hsv[2] *= value
        value *= 0.9
        color_rgb = color.hsv2rgb(color_hsv)

        ax.axis('off')
        # x = offset * np.random.uniform(-2, 2, int(n_points))
        x = np.random.normal(0, offset * 2, int(n_points))
        y = 1 * np.sinc(x ** 2)
        ax.scatter(x, y / offset, color=color_rgb, s=2)
        # ax.set_aspect(5)
        offset += 3
        ax.set_xlim(-100, 100)
    if save_fig:
        plt.savefig('dotted_lines')
    plt.show()


if __name__ == '__main__':
    run_1()

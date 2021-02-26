import matplotlib.pyplot as plt
import numpy as np
from skimage import color
import randomcolor
from scipy import signal
from scipy.signal import chirp


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


def run_2(save_fig=False):
    n_lines = 3
    n_points = 800
    x_min = -20
    x_max = 20

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
        # x = np.random.normal(0, offset * 2, int(n_points))
        mode = np.random.uniform(x_min, x_max)
        x = np.random.triangular(x_min, mode, x_max, size=int(n_points))

        y = 1 * np.sinc(x ** 2)
        ax.scatter(x, y / offset, color=color_rgb, s=1.5)
        ax.set_aspect(15)
        offset += 3
        ax.set_xlim(x_min, x_max)
    fig.set_figheight(5)
    fig.set_figwidth(8)

    if save_fig:
        plt.savefig('dotted_lines')
    plt.show()


def wavelet_ricker_series():
    n_lines = 15
    n_points = 100
    max_offset = 800

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
        for _ in range(5):
            x_offset = np.random.uniform(0, max_offset)
            a = np.random.uniform(1, 5)
            n_points /= 2.
            color_rgb = np.array(rand_color.generate(hue=hue,
                                                     luminosity=luminosity,
                                                     format_='Array_rgb')) / 256.
            color_rgb = color_rgb[0]
            color_hsv = color.rgb2hsv(color_rgb)
            color_hsv[2] *= 0.4 + 0.6 * x_offset / max_offset
            color_rgb = color.hsv2rgb(color_hsv)

            ax.axis('off')
            n_points = 500
            sample_frac = 0.1
            mode = np.random.uniform(0, n_points)
            x = np.random.triangular(0, mode, n_points, size=int(n_points * sample_frac))
            x = [int(i) for i in x]
            y = signal.ricker(int(n_points), 25)
            y = y[x]

            ax.scatter(x_offset + np.array(x), 90 * y, color=color_rgb, s=10.0, alpha=0.99)
            # ax.set_aspect(2)
            offset += 3
            # ax.set_xlim(0, 500)
    fig.set_figheight(5)
    # fig.set_figwidth(4)

    plt.show()


def weird_test():

    n_lines = 4

    hue = np.random.choice(['blue', 'purple', 'red'])

    luminosity = None
    fig, axes = plt.subplots(n_lines, frameon=True)
    rand_color = randomcolor.RandomColor()
    color_rgb = np.array(rand_color.generate(hue=hue,
                                             luminosity=luminosity,
                                             format_='Array_rgb')) / 256.
    color_rgb_background = color_rgb[0]
    color_rgb = np.array(rand_color.generate(hue=hue,
                                             luminosity=luminosity,
                                             format_='Array_rgb')) / 256.
    color_rgb = color_rgb[0]
    color_hsv = color.rgb2hsv(color_rgb_background)
    color_hsv[2] *= 0.4 + 0.6 * x_offset / max_offset
    color_rgb_line = color.hsv2rgb(color_hsv)

    t = np.linspace(0, 10, 1500)
    w = chirp(t, f0=6, f1=1, t1=10, method='linear')

    axes[0].plot(t, w, color=color_rgb)
    axes[0].set_facecolor(color_rgb_background)

    widths = np.arange(1, 31)
    cwtmatr = signal.cwt(w, signal.ricker, widths)
    axes[1].imshow(cwtmatr, extent=[-1, 1, 1, 31], cmap='PRGn', aspect='auto',
                   vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())

    axes[2].plot(cwtmatr[0, :])

    for item in cwtmatr:
        color_rgb = np.array(rand_color.generate(hue=hue,
                                                 luminosity=luminosity,
                                                 format_='Array_rgb')) / 256.
        axes[3].plot(item, color=color_rgb[0])


def holiday():
    n_lines = 5

    hue = np.random.choice(['blue', 'purple', 'red'])

    luminosity = None
    fig, axes = plt.subplots(1, n_lines)
    rand_color = randomcolor.RandomColor()
    background_color = np.array(rand_color.generate(hue=hue,
                                                    luminosity=luminosity,
                                                    format_='Array_rgb')) / 256.
    plt.rcParams["figure.facecolor"] = background_color[0]

    for axes_index in range(n_lines):

        axes[axes_index].axis('off')
        rand_color = randomcolor.RandomColor()
        color_rgb = np.array(rand_color.generate(hue=hue,
                                                 luminosity=luminosity,
                                                 format_='Array_rgb')) / 256.
        color_rgb_background = color_rgb[0]
        color_rgb = np.array(rand_color.generate(hue=hue,
                                                 luminosity=luminosity,
                                                 format_='Array_rgb')) / 256.
        color_rgb = color_rgb[0]
        color_hsv = color.rgb2hsv(color_rgb_background)
        color_hsv[2] *= 0.95

        t = np.linspace(0, 10, 1500)
        w = chirp(t, f0=6, f1=1, t1=10, method='linear')

        axes[0].plot(t, w, color=color_rgb)
        axes[0].set_facecolor(color_rgb_background)

        top_freq = np.random.uniform(30, 80)
        widths = np.arange(1, top_freq)
        cwtmatr = signal.cwt(w, signal.ricker, widths)

        for item in cwtmatr:
            color_rgb = np.array(rand_color.generate(hue=hue,
                                                     luminosity=luminosity,
                                                     format_='Array_rgb')) / 256.
            axes[axes_index].plot(item, range(len(item)), color=color_rgb[0])
            x_min = np.random.uniform(np.min(item), np.max(item) * 0.8)
            height = np.max(item) - np.min(item)
            x_max = x_min + 0.25 * height
            axes[axes_index].set_xlim(x_min, x_max)


def fun_squiggles():
    n_lines = 5

    hue = np.random.choice([
        'blue',
        'purple',
        'red',
        'green'
    ])

    luminosity = None
    fig, axes = plt.subplots(1, n_lines)
    rand_color = randomcolor.RandomColor()
    background_color = np.array(rand_color.generate(hue=hue,
                                                    luminosity=luminosity,
                                                    format_='Array_rgb')) / 256.
    plt.rcParams["figure.facecolor"] = background_color[0]

    for axes_index in range(n_lines):

        axes[axes_index].axis('off')
        rand_color = randomcolor.RandomColor()
        color_rgb = np.array(rand_color.generate(hue=hue,
                                                 luminosity=luminosity,
                                                 format_='Array_rgb')) / 256.
        color_rgb_background = color_rgb[0]
        color_rgb = np.array(rand_color.generate(hue=hue,
                                                 luminosity=luminosity,
                                                 format_='Array_rgb')) / 256.
        color_rgb = color_rgb[0]
        color_hsv = color.rgb2hsv(color_rgb_background)
        color_hsv[2] *= 0.95

        t = np.linspace(0, 10, 1500)
        w = chirp(t, f0=6, f1=1, t1=10, method='linear')

        axes[0].plot(t, w, color=color_rgb)
        axes[0].set_facecolor(color_rgb_background)

        top_freq = np.random.uniform(30, 80)
        widths = np.arange(1, top_freq)
        cwtmatr = signal.cwt(w, signal.ricker, widths)

        for item in cwtmatr:
            color_rgb = np.array(rand_color.generate(hue=hue,
                                                     luminosity=luminosity,
                                                     format_='Array_rgb')) / 256.
            axes[axes_index].plot(item, range(len(item)), color=color_rgb[0])
            height = np.max(item) - np.min(item)
            x_min = np.random.uniform(np.min(item), np.max(item) * 0.8)
            if axes_index == 0:
                x_min = np.min(item)  # + 0.1 * height

            if axes_index == n_lines - 1:
                x_max = np.max(item)  # - 0.1 * height
            x_max = x_min + 0.25 * height
            axes[axes_index].set_xlim(x_min, x_max)


if __name__ == '__main__':
    run_1()

import matplotlib.pyplot as plt
import randomcolor
import numpy as np
from scipy.signal import cwt, ricker


def run_1():
    fig, ax = plt.subplots()
    fig.patch.set_visible(False)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor((0.6, 0.65, 0.67))
    line_nums = 10

    x = np.linspace(0, 100, 1000)

    for line_num in range(line_nums):
        freq = np.random.uniform(1.0, 5.0)
        offset = np.random.uniform(1.0, 2.0)
        y = line_num * 1.0 + np.sin(freq * (x - offset))
        plt.plot(x, y)

    plt.show()


def run_2(plot_points=False):
    # matplotlib Agnes Martin
    fig, ax = plt.subplots()
    fig.patch.set_visible(False)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor((0.2, 0.23, 0.24))
    line_nums = 10

    x = np.linspace(0, 50, 1000)

    rand_color = randomcolor.RandomColor()

    for line_num in range(line_nums):
        random_color = np.array(rand_color.generate(hue="blue", count=1, format_='Array_rgb')) / 256.
        freq = np.random.uniform(1.0, 5.0)
        offset = np.random.uniform(1.0, 2.0)
        y = np.sin(freq * (x - offset))

        widths = range(1, 50)
        cwt_output = cwt(y, ricker, widths)
        for output in cwt_output:
            if plot_points:
                ax.scatter(x, line_num * 1.0 + output, color=random_color[0], s=0.3)
            else:
                ax.plot(x, line_num * 1.0 + output, c=random_color[0], linewidth=0.1)

    x_lims_current = ax.get_xlim()
    y_lims_current = ax.get_ylim()

    lower_xlim = np.random.uniform(*x_lims_current)
    x_axes_length = np.random.uniform(2, 10)
    upper_xlim = np.min([lower_xlim + x_axes_length, x_lims_current[1]])
    ax.set_xlim(lower_xlim, upper_xlim)

    lower_ylim = np.random.uniform(*y_lims_current)
    y_axes_length = np.random.uniform(2, 10)
    upper_ylim = np.min([lower_ylim + y_axes_length, y_lims_current[1]])
    ax.set_ylim(lower_ylim, upper_ylim)

    plt.show()


if __name__ == '__main__':
    run_2(True)

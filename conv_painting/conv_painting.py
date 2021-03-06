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


def run_2():
    fig, ax = plt.subplots()
    fig.patch.set_visible(False)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor((0.6, 0.65, 0.67))
    line_nums = 10

    x = np.linspace(0, 100, 1000)

    rand_color = randomcolor.RandomColor()

    for line_num in range(line_nums):
        random_color = np.array(rand_color.generate(hue="blue", count=1, format_='Array_rgb')) / 256.
        freq = np.random.uniform(1.0, 5.0)
        offset = np.random.uniform(1.0, 2.0)
        y = np.sin(freq * (x - offset))

        widths = range(1, 5)
        cwt_output = cwt(y, ricker, widths)
        for output in cwt_output:
            plt.plot(x, line_num * 1.0 + output, c=random_color[0])
    plt.show()

if __name__ == '__main__':
    run_2()

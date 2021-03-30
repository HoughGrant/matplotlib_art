import matplotlib.pyplot as plt
import numpy as np
from skimage import color
import randomcolor

def run_1():
    fig = plt.figure(constrained_layout=False, frameon=False)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor((0.37, 0.37, 0.35))

    rand_color = randomcolor.RandomColor()

    for index, total_points in enumerate([50, 80, 120]):

        main_rgb_color = np.array(rand_color.generate(hue="blue", count=1, format_='Array_rgb')) / 256.
        main_rgb_color = main_rgb_color[0]

        x = np.linspace(0, 3, total_points)
        for _ in range(500):
            if index == 0:
                y = np.random.uniform(-30, 30) * x ** 2 + np.random.uniform(-15, 15) * x ** 1 + np.random.uniform(-1, 1)
            else:
                y = np.random.uniform(-25, 20) * x ** 2 + np.random.uniform(-10, 10) * x ** 1 + np.random.uniform(-10, 10)

            max = 20
            err = np.random.uniform(5, max) * np.sin(x * np.random.uniform(5, max))
            for _ in range(5):
                err += np.random.uniform(5, max) * np.sin(x * np.random.uniform(5, max))
                max *= 0.4
                max = np.max([max, 20])

            if np.random.rand() < 0.5:
                y = y[::-1]

            tmp_hsv_color = color.rgb2hsv(main_rgb_color)
            tmp_hsv_color[2] *= np.random.uniform(0.4, 0.7)
            ecolor = color.hsv2rgb(tmp_hsv_color)

            if index == 0:
                elinewidth = 100
                alpha = 0.05
            else:
                elinewidth = 5
                alpha = 1 - np.random.power(1)

            ax.errorbar(x, y, err, linewidth=0, ecolor=ecolor, elinewidth=elinewidth, barsabove=True,
                        errorevery=total_points // 40, snap=True, alpha=alpha)

    plt.show()


if __name__ == '__main__':
    run_1()

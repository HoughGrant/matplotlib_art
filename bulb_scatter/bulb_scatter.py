import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from skimage import color
import randomcolor


def run_1():
    fig, ax = plt.subplots()

    point = [0, 0]
    for _ in range(10):
        circle = mpatches.Circle(point, radius=1, transform=ax.transData)
        ax.add_patch(circle)
        # point += np.random.uniform(-2.5, 2, 2)
        point += np.random.randint(-2, 2, 2)
        print(point)

    ax.axis('equal')
    ax.axis('off')
    plt.tight_layout()
    plt.show()


def sigmoid(x, b0, b1):
    return 1 / (1 + np.exp(-(b0 + b1 * x)))


def run_2():

    fig = plt.figure(constrained_layout=False, frameon=False)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor((0.07, 0.07, 0.068))
    rand_color = randomcolor.RandomColor()

    tmp_color = np.array(rand_color.generate(hue='blue',
                                             luminosity="light",
                                             count=1,
                                             format_='Array_rgb')) / 256.
    x = np.linspace(-0.05, 0.075, 100)
    b0 = 0

    all_colors_hsv = color.rgb2hsv(tmp_color[0])
    all_colors_hsv[2] = 1.0
    b1_prev = None
    b0_prev = None

    for k in range(1, 20)[::-1]:

        k *= 2

        b1_prev = None
        b0_prev = None
        for b1 in np.logspace(-2, 0.5, 10)[::-1]:
            if b1_prev is None:
                b1_prev = b1
                b0_prev = b0
                continue

            all_colors_hsv[2] *= 0.99
            all_colors_rgb = color.hsv2rgb(all_colors_hsv)

            b0 += np.random.uniform(-0.2, 0.2)
            ax.plot(sigmoid(x, b0, b1), x, color=all_colors_rgb, alpha=0.4, linewidth=0.2)
            if np.random.random() < 1.0:
                ax.fill_betweenx(x, sigmoid(x, b0_prev, b1_prev), sigmoid(x, b0, b1), color=all_colors_rgb, alpha=0.2)
            b1_prev = b1
            b0_prev = b0

    all_colors_hsv[2] = 1.0
    for b1 in np.logspace(1, 2.5, 100)[::-1]:
        if b1_prev is None:
            b1_prev = b1
            b0_prev = b0
            continue

        all_colors_hsv[2] *= 0.99
        all_colors_rgb = color.hsv2rgb(all_colors_hsv)

        b0 += np.random.uniform(-0.2, 0.2)
        ax.plot(sigmoid(x, b0, b1), x, color=all_colors_rgb, alpha=0.6, linewidth=0.2)
        ax.fill_betweenx(x, sigmoid(x, b0, b1), sigmoid(x, b0_prev, b1_prev), color=all_colors_rgb, alpha=0.5)
        b1_prev = b1
        b0_prev = b0
    plt.show()


if __name__ == '__main__':
    # run_1()
    run_2()

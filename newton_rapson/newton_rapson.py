import matplotlib.pyplot as plt
import numpy as np
import randomcolor
from skimage import color
from scipy.optimize import newton


def run_1():

    def iter_func_pretty(guess):
        guess_updated = (0.001 * guess ** 3 + 1) / (0.1 * guess ** 2 - 53)
        return guess_updated

    def objective_func(x):
        p = np.poly1d([np.random.uniform(-1, 1),
                       # np.random.uniform(-1, 1),
                       # np.random.uniform(-1, 1),
                       np.random.uniform(-1, 1)],
                      True)
        return p(x)
        # return np.sin(x)

    def iter_func(guess):
        # guess_updated = (2 * guess ** 3 + 1) / (3 * guess ** 2 - 3)
        guess_updated = newton(objective_func, guess, maxiter=1, tol=1e10)
        # guess_updated = np.sin(guess * 5) / (5 * np.cos(guess * 5))
        # guess_updated = np.sin(guess * 5) / (5 * np.cos(guess * 5))
        return guess_updated

    def test_iterator(start_guess, total_iters):
        path = [start_guess]
        for _ in range(total_iters):
            path.append(iter_func(path[-1]))
        return np.array(path)

    fig, ax = plt.subplots()
    ax = fig.add_axes([0, 0, 1, 1])

    rand_color = randomcolor.RandomColor()
    hue = np.random.choice(['blue', 'purple', 'red'])
    luminosity = None
    background_color = np.array(rand_color.generate(hue=hue,
                                                    luminosity=luminosity,
                                                    format_='Array_rgb')) / 256.
    color_hsv = color.rgb2hsv(background_color[0])
    color_hsv[2] *= 0.4
    color_rgb = color.hsv2rgb(color_hsv)
    ax.set_facecolor(color_rgb)

    value_modifier = 0.5
    for tmp_guess in np.linspace(-1, 1, 100):
        example_path = test_iterator(tmp_guess, 1000)

        line_color_rgb = np.array(rand_color.generate(hue=hue,
                                                  luminosity=luminosity,
                                                  format_='Array_rgb')) / 256.
        line_color_hsv = color.rgb2hsv(line_color_rgb[0])
        value_modifier *= 1.01
        value_modifier = np.min([value_modifier, 0.95])
        line_color_hsv[2] = value_modifier
        line_color_rgb = color.hsv2rgb(line_color_hsv)

        alpha = np.random.uniform(1 - value_modifier, 0.8)
        linewidth = 5 * (1 - alpha)

        ax.plot(example_path, c=line_color_rgb, alpha=alpha, linewidth=linewidth)
        # ax.semilogy(example_path, c=line_color_rgb, alpha=alpha, linewidth=linewidth)
        # ax.scatter(range(len(example_path)), example_path)
    plt.show()


if __name__ == '__main__':
    run_1()

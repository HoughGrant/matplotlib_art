import matplotlib.pyplot as plt
import randomcolor
import numpy as np
from skimage import color


def run_1():
    fig, ax = plt.subplots()
    ax = fig.add_axes([0, 0, 1, 1])
    linewidth = 5
    num_sins = 1
    n_lines = 50
    line_height = 10
    background_hsv_aug = 0.5
    shade_hsv_aug = 0.5

    max_dashes = 20
    max_dash_style = 10
    min_dash_style = 2
    sin_freq_min = 2
    sin_freq_max = 4

    hue = np.random.choice(['blue', 'purple', 'red'])
    luminosity = None

    rand_color = randomcolor.RandomColor()
    all_colors = np.array(rand_color.generate(hue=hue,
                                              luminosity=luminosity,
                                              count=n_lines,
                                              format_='Array_rgb')) / 256.

    background_color = np.array(rand_color.generate(hue=hue,
                                                    luminosity=luminosity,
                                                    format_='Array_rgb')) / 256.
    color_hsv = color.rgb2hsv(background_color[0])
    color_hsv[2] *= background_hsv_aug
    color_rgb = color.hsv2rgb(color_hsv)
    ax.set_facecolor(color_rgb)
    for index in range(n_lines):
        x = [index / n_lines, index / n_lines]
        y = [0, line_height]

        total_dashes = np.random.randint(1, max_dashes)
        dash_style = np.random.randint(min_dash_style, max_dash_style, max_dashes)
        dash_style = dash_style[:total_dashes]

        color_rgb_orig = all_colors[index]
        color_hsv = color.rgb2hsv(color_rgb_orig)
        color_hsv[2] *= shade_hsv_aug
        color_rgb = color.hsv2rgb(color_hsv)
        ax.plot(x, y, color=color_rgb, linestyle=(0, dash_style),
                linewidth=linewidth)

        for _ in range(num_sins):
            sin_offset = np.random.uniform(1, 0.8 * y[1])
            sin_freq = np.random.uniform(sin_freq_min, sin_freq_max)
            y[1] = sin_offset + np.sin(x[0] * sin_freq)
            ax.plot(np.array(x) - 0.005, y, color=color_rgb_orig, linestyle=(0, dash_style),
                    linewidth=linewidth, alpha=0.9)

    plt.show()


if __name__ == '__main__':
    run_1()

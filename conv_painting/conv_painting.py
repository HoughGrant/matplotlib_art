import matplotlib.pyplot as plt
import randomcolor
import numpy as np
from scipy.signal import cwt, ricker
from sklearn.metrics import mean_squared_error
from skimage.transform import resize
from skimage.io import imread
import io


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


def run_2(plot_points=False, random_zoom=False):
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

    if random_zoom:
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


def figure_to_ndarray(fig, image_shape=None):
    io_buf = io.BytesIO()
    fig.savefig(io_buf, format='raw')
    io_buf.seek(0)
    img_arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                         newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
    if image_shape:
         img_arr = resize(img_arr, (image_shape[0], image_shape[1], img_arr.shape[2]))
    io_buf.close()
    return img_arr[:, :, :3]

def fig_random_zoom(ax):
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


def add_line(ax, freq, offset, widths_max, color, alpha, linewidth):
    added_lines = []
    line_nums = 10
    line_num = np.random.randint(1, line_nums)
    x = np.linspace(20, 40, 1000)
    y = np.sin(freq * (x - offset))

    widths = range(1, widths_max)
    cwt_output = cwt(y, ricker, widths)
    for output in cwt_output:
        added_lines.append(ax.plot(x, line_num * 1.0 + output, c=color, linewidth=linewidth, alpha=alpha))
    return added_lines


def compare_target_and_predict(target_image, predict_image):
    mse = np.mean((target_image - predict_image)**2)
    return mse


def conv_painting_hill_climbing(image_url, plot_points=False, random_zoom=False, target_image_shape=1024, total_iters=100):
    mse_per_iteration = []

    target_image = imread(image_url)
    target_image_resized = resize(target_image, (target_image_shape, target_image_shape, 3))
    target_image_resized_average_color = np.average(target_image_resized, axis=(0, 1))

    fig, ax = plt.subplots()
    fig.patch.set_visible(False)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(target_image_resized_average_color)

    predict_image = figure_to_ndarray(fig, (target_image_shape, target_image_shape))
    mse_init = compare_target_and_predict(target_image_resized, predict_image)

    for i in range(total_iters):
        plot_flag = False
        freq = np.random.uniform(1.0, 5.0)
        offset = np.random.uniform(1.0, 2.0)
        widths_max = np.random.randint(2, 50)
        alpha = np.random.uniform(0.1, 0.5)
        linewidth = np.random.uniform(0.1, 2.0)

        if np.random.random() > 0.5:
            rand_color = randomcolor.RandomColor()
            random_color = np.array(rand_color.generate(count=1, format_='Array_rgb')) / 256.
            color = random_color[0]
        else:
            rand_x = np.random.randint(target_image.shape[0])
            rand_y = np.random.randint(target_image.shape[1])
            color = np.array(target_image[rand_x, rand_y] / 256.)

        # added_lines = add_line(ax, freq, offset, widths_max, color, alpha,
        #                        linewidth)
        # predict_image = figure_to_ndarray(fig, (target_image_shape, target_image_shape))
        # mse_init = compare_target_and_predict(target_image_resized, predict_image)
        # for line in added_lines:
        #     line.pop(0).remove()

        for k in range(10):
            freq_update = freq * np.random.uniform(0.9, 1.1)
            offset_update = offset * np.random.uniform(0.9, 1.1)
            color_update = color * np.random.uniform(0.9, 1.1)
            color_update = np.array([np.min([1.0, color]) for color in color_update])
            alpha_update = alpha * np.random.uniform(0.9, 1.1)
            alpha_update = np.min([1.0, alpha_update])
            linewidth_update = linewidth * np.random.uniform(0.9, 1.1)

            added_lines = add_line(ax, freq_update, offset_update, widths_max, color_update, alpha_update,
                                   linewidth_update)
            predict_image = figure_to_ndarray(fig, (target_image_shape, target_image_shape))
            mse_update = compare_target_and_predict(target_image_resized, predict_image)
            for line in added_lines:
                line.pop(0).remove()
            if mse_update < mse_init:
                print(k, mse_init, mse_update)
                freq = freq_update
                offset = offset_update
                color = color_update
                alpha = alpha_update
                linewidth = linewidth_update
                mse_init = mse_update
                plot_flag = True

        if plot_flag:
            _ = add_line(ax, freq, offset, widths_max, color, alpha, linewidth)
            print('Iter ', i, mse_init)
        mse_per_iteration.append(mse_init)

    final_image = figure_to_ndarray(fig, (target_image.shape[0], target_image.shape[1]))
    plt.show()

    plt.figure()
    plt.imshow(final_image)
    plt.show()
    plt.figure()
    plt.imshow(target_image)
    plt.show()

    plt.figure()
    plt.plot(mse_per_iteration)
    plt.show()

    print(5)

if __name__ == '__main__':
    conv_painting_hill_climbing('https://images-na.ssl-images-amazon.com/images/I/71lix6%2BVfWL._SY450_.jpg',
                                total_iters=1000)

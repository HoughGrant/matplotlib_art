import matplotlib.pyplot as plt
import numpy as np


def simple_lines(save_fig=False, num_lines=20):
    """
    I like the default color pallet here
    :param save_fig:
    :return:
    """
    fig = plt.figure(frameon=True)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor((0.2, 0.25, 0.23))

    x, y = np.linspace(0, 100, 10), np.zeros(10)
    for i in range(num_lines):
        y_offset = np.random.uniform(-1, 1)
        x_offset = np.random.uniform(-0.1, 0.1)
        ax.plot(x + x_offset, y + y_offset, linestyle='-', linewidth=1.5, color=(1, 1, 1, 0.5))

    if save_fig:
        plt.savefig('simple_lines')
    plt.show()


def simple_forest(save_fig=False, num_lines=200):
    """
    I like the default color pallet here
    :param save_fig:
    :return:
    """
    fig = plt.figure(frameon=True)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor((0.2, 0.25, 0.23))

    # vertical lines
    num_vert = num_lines
    x, y = np.zeros(num_vert), np.linspace(0, 10, num_vert)
    for i in range(num_vert):
        x_offset = np.random.uniform(-1, 1)
        line_size = np.random.exponential(10)
        color = line_size * np.array([0.05, 0.3, 0.1])
        color /= np.max(color)
        color *= 0.6
        alpha = [np.random.uniform(0, 0.2) if line_size>3 else np.random.uniform(0.3, 0.7)]
        color = np.append(color, alpha)
        ax.plot(x + x_offset, y, linestyle='-', linewidth=line_size, color=color)

    num_vert_brown = num_lines // 10
    x, y = np.zeros(num_vert), np.linspace(0, 10, num_vert)
    for i in range(num_vert_brown):
        x_offset = np.random.uniform(-1, 1)
        line_size = np.random.exponential(20)
        color = line_size * np.array([0.05, 0.07, 0.1])
        color /= np.max(color)
        color *= 0.1
        alpha = [np.random.uniform(0, 0.2) if line_size > 40 else np.random.uniform(0.2, 0.3)]
        color = np.append(color, alpha)
        ax.plot(x + x_offset, y, linestyle='-', linewidth=line_size, color=color)

    if save_fig:
        plt.savefig('simple_forest')
    plt.show()


def watermellon_roll(save_fig=False, num_lines=200):
    """
    I like the default color pallet here
    :param save_fig:
    :return:
    """
    fig = plt.figure(frameon=True)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor((0.2, 0.25, 0.23))

    # vertical lines
    num_vert = num_lines
    x, y = np.zeros(num_vert), np.linspace(0, 10, num_vert)
    for i in range(num_vert):
        x_offset = np.random.uniform(-1, 1)
        line_size = np.random.exponential(10)
        color = line_size * np.array([0.05, 0.3, 0.1])
        color /= np.max(color)
        color *= 0.6
        alpha = [np.random.uniform(0, 0.2) if line_size>3 else np.random.uniform(0.3, 0.7)]
        color = np.append(color, alpha)
        ax.plot(x + x_offset, y, linestyle='-', linewidth=line_size, color=color)

    num_vert_brown = num_lines // 10
    x, y = np.zeros(num_vert), np.linspace(0, 10, num_vert)
    for i in range(num_vert_brown):
        x_offset = np.random.uniform(-1, 1)
        line_size = np.random.exponential(20)
        color = line_size * np.array([0.05, 0.07, 0.1])
        color /= np.max(color)
        color *= 0.1
        alpha = [np.random.uniform(0, 0.2) if line_size > 40 else np.random.uniform(0.2, 0.3)]
        color = np.append(color, alpha)
        ax.plot(x + x_offset, y, linestyle='-', linewidth=line_size, color=color)

    # horizontal lines
    x, y = np.linspace(0, 10, 10), np.zeros(10)
    # for i in range(num_lines):
    #     y_offset = np.random.uniform(-1, 1)
    #     x_offset = np.random.uniform(-0.1, 0.1)
    #     ax.plot(x + x_offset, y + y_offset, linestyle='-', linewidth=1.5, color=(1, 1, 1, 0.5))

    plt.show()

    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()

    fig = plt.figure(frameon=True)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor((0.2, 0.25, 0.23))
    data_mod = [color for color in data]
    data_mod[-200:] = np.roll(data_mod[-200:], 50, axis=0)
    data_mod[:150] = np.roll(data_mod[:150], 60, axis=0)
    data_mod[100:150] = np.roll(data_mod[100:150], 50, axis=2)
    data_mod[105:138] = np.roll(data_mod[105:138], 75, axis=1)
    data_mod[125:130] = np.roll(data_mod[125:130], 50, axis=1)

    data_mod[:][105:138] = np.roll(data_mod[:][105:138], 75, axis=1)
    # data_mod[:][105:138] = np.roll(data_mod[:][105:138], 75, axis=0)
    ax.imshow(data_mod)
    plt.show()
    if save_fig:
        plt.savefig('watermellon_roll')
    # plt.show()


def watermellon_roll_random(save_fig=False, num_lines=200):
    """
    I like the default color pallet here
    :param save_fig:
    :return:
    """
    fig = plt.figure(frameon=True)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor((0.2, 0.25, 0.23))

    # vertical lines
    num_vert = num_lines
    x, y = np.zeros(num_vert), np.linspace(0, 10, num_vert)
    for i in range(num_vert):
        x_offset = np.random.uniform(-1, 1)
        line_size = np.random.exponential(10)
        color = line_size * np.array([0.05, 0.3, 0.1])
        color /= np.max(color)
        color *= 0.6
        alpha = [np.random.uniform(0, 0.2) if line_size>3 else np.random.uniform(0.3, 0.7)]
        color = np.append(color, alpha)
        ax.plot(x + x_offset, y, linestyle='-', linewidth=line_size, color=color)

    num_vert_brown = num_lines // 10
    x, y = np.zeros(num_vert), np.linspace(0, 10, num_vert)
    for i in range(num_vert_brown):
        x_offset = np.random.uniform(-1, 1)
        line_size = np.random.exponential(20)
        color = line_size * np.array([0.05, 0.07, 0.1])
        color /= np.max(color)
        color *= 0.1
        alpha = [np.random.uniform(0, 0.2) if line_size > 40 else np.random.uniform(0.2, 0.3)]
        color = np.append(color, alpha)
        ax.plot(x + x_offset, y, linestyle='-', linewidth=line_size, color=color)

    plt.show()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()

    fig = plt.figure(frameon=True)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor((0.2, 0.25, 0.23))

    data_mod = [color for color in data]
    data_mod[-200:] = np.roll(data_mod[-200:], 50, axis=0)
    data_mod[:150] = np.roll(data_mod[:150], 60, axis=0)
    width = np.random.uniform(5, 20)
    start_height = np.random.uniform(50, np.array(data_mod).shape[1] - width - 2)

    for k, axis, amount in zip((0.8, 0.6, 0.51), (2, 1, 1), (50, 70, 50)):
        start = int((1 - k) * start_height)
        end = int(k * (start_height + width))
        print(start, end)
        data_mod[start:end] = np.roll(data_mod[start:end], amount, axis=axis)

    ax.imshow(data_mod)
    plt.show()

    ax.imshow(data_mod)
    plt.show()
    if save_fig:
        plt.savefig('watermellon_roll_random')


if __name__ == '__main__':
    # lines()
    # simple_forest()
    # watermellon_roll()
    watermellon_roll_random()
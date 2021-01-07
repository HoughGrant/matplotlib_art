import matplotlib.pyplot as plt
import numpy as np
import random


def run_1(num_points=10, lower_bound=0, upper_bound=100):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor((0.3, 0.2, 0.2))
    ax.axis('off')
    points = np.random.randint(lower_bound, upper_bound, (num_points, 2))
    for point in points:
        marker = np.random.randint(0, 10)
        ax.scatter(*point, marker='$' + str(marker) + '$', color=(0.2, 0.0, 0.0, 0.2))
    plt.show()


def run_1_rand_alpha(num_points=10, lower_bound=0, upper_bound=100):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor((0.3, 0.2, 0.2))
    ax.axis('off')
    points = np.random.randint(lower_bound, upper_bound, (num_points, 2))
    for point in points:
        marker = np.random.randint(0, 10)
        alpha = np.random.pareto(10)
        alpha = np.min((alpha, 1))
        # alpha = uniform(0.1, 0.4)
        ax.scatter(*point, marker='$' + str(marker) + '$', color=(0.2, 0.0, 0.0, 0.2), alpha=alpha)
    plt.show()


def run_2(lower_bound=0, upper_bound=100):
    fig, ax = plt.subplots(nrows=3, ncols=1)
    for i, num_points in enumerate([25, 100, 2000]):
        points = np.random.randint(lower_bound, upper_bound, (num_points, 2))
        for point in points:
            marker = np.random.randint(0, 10)
            ax[i].scatter(*point, marker='$' + str(marker) + '$', color=(0.2, 0.0, 0.0, 0.2))
    plt.show()


def run_2_alpha(lower_bound=0, upper_bound=100):
    fig, ax = plt.subplots(nrows=3, ncols=1)
    for i, num_points in enumerate([25, 100, 2000]):
        points = np.random.randint(lower_bound, upper_bound, (num_points, 2))
        for point in points:
            marker = np.random.randint(0, 10)
            alpha = np.random.pareto(5 + i)
            alpha = np.min((alpha, 1))
            ax[i].scatter(*point, marker='$' + str(marker) + '$', color=(0.2, 0.0, 0.0, 0.2), alpha=alpha)
    plt.show()


def run_2_alpha_too_much(lower_bound=0, upper_bound=100):
    fig, ax = plt.subplots(nrows=9, ncols=1)
    potential_points = [25, 100, 200, 40, 20, 0, 500, 0, 0]
    random.shuffle(potential_points)
    for i, num_points in enumerate(potential_points):
        points = np.random.randint(lower_bound, upper_bound, (num_points, 2))
        for point in points:
            marker = np.random.randint(0, 10)
            alpha = np.random.pareto(5 + i)
            alpha = np.min((alpha, 1))
            ax[i].scatter(*point, marker='$' + str(marker) + '$', color=(0.2, 0.0, 0.0, 0.2), alpha=alpha, s=10)
    plt.show()

if __name__ == '__main__':
    # run_1(100)
    # run_1_rand_alpha(1000)
    # run_2()
    # run_2_alpha()
    run_2_alpha_too_much()

import matplotlib.pyplot as plt
import numpy as np
import random

import randomcolor


def check_if_all_nums_in_draw(draw, highest_num):
    for num in range(1, highest_num+1):
        if num not in draw:
            return False
    return True


def problem1():
    """
    Urn contains N balls, N1 color 1, N2 color 2, ... Nk color k. Draw m balls w/o replacement, what is prob that we have at
    least one of each color? Given k=5, Ni = 10 for all i, how many do we need to draw to have at least 90% prob of getting
    a full set?
    """
    n_i = 10
    k = 5
    num_samples = 1000
    total_draws = 50

    plt.figure()
    for num_samples in [100, 1000, 10000]:
        experiment_results = []
        for samples in range(num_samples):
            # N = np.random.randint(1, k + 1, n_i * k)
            N = np.array([[i] * n_i for i in range(1, k+1)]).flatten()
            random.shuffle(N)
            experiment_results_for_sample = []
            for n_draws in range(1, total_draws + 1):
                draw = N[:n_draws]
                experiment_result = check_if_all_nums_in_draw(draw, k)
                experiment_results_for_sample.append(experiment_result)
            experiment_results.append(experiment_results_for_sample)
        experiment_results = np.array(experiment_results)

        plt.plot(range(1, total_draws + 1), np.sum(experiment_results, axis=0)/num_samples, label=num_samples)

    plt.plot([1, total_draws+1], [0.9, 0.9])
    plt.xlabel('Total Draws')
    plt.ylabel('Probability')
    plt.xlim(1, total_draws)
    plt.legend()
    plt.show()


def problem2():
    """
    Urn has 50 balls, 4 colors, only one is the fourth color. What is prob we don't pull the fourth color on the first 20
    draws?
    """
    k = 4
    total_draws = 20
    total_balls = 50

    plt.figure()
    for _ in range(50):
        for num_samples in [10000]:
            experiment_results = []
            for samples in range(num_samples):
                N = np.random.randint(1, k, total_balls - 1)
                N = np.append(N, k)
                N = np.array(N).flatten()
                random.shuffle(N)
                draw = N[:total_draws]
                experiment_result = np.any(draw == 4)
                experiment_results.append(experiment_result)
            plt.plot(np.cumsum(experiment_results) / np.arange(1, num_samples + 1))
        old_result = experiment_results[:]

    plt.xlabel('Total Draws')
    plt.ylabel('Probability')
    plt.show()


def for_fun():
    """
    Urn has 50 balls, 4 colors, only one is the fourth color. What is prob we don't pull the fourth color on the first 20
    draws?
    """
    k = 10
    total_draws = 35
    total_balls = 40
    n_experiments = 100
    old_result = None

    rand_color = randomcolor.RandomColor()
    fig = plt.figure(constrained_layout=False, frameon=False)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor((0.07, 0.07, 0.05))

    # for total_draws, color in zip([20, 25, 30], ['red', 'red', 'red']):
    # for total_draws, color in zip([20, 25, 30], ['purple', 'yellow', 'purple']): # mardi gras argyle
    # for total_draws, color in zip([5, 25, 27, 23, 40], ['purple', 'purple', 'blue', 'blue', 'purple']):
    for total_draws, color in zip([20, 3, 5, 10, 35], ['blue', 'red', 'blue', 'purple', 'blue']): # this one is good
        for _ in range(n_experiments):
            for num_samples in [10000]:
                experiment_results = []
                for samples in range(num_samples):
                    N = np.random.randint(1, k, total_balls - 1)
                    N = np.append(N, k)
                    N = np.array(N).flatten()
                    random.shuffle(N)
                    draw = N[:np.random.randint(total_draws - 3, total_draws + 3)]
                    experiment_result = np.any(draw == k)
                    experiment_results.append(experiment_result)
                if old_result:
                    if np.random.uniform(0, 1) > 0.8:
                        luminosity = None
                        if color == 'green':
                            luminosity = 'bright'
                        if color == 'yellow':
                            luminosity = 'dark'
                        tmp_rgb_color = np.array(rand_color.generate(
                            hue=color, luminosity=luminosity, count=1, format_='Array_rgb')) / 256.
                        tmp_rgb_color = tmp_rgb_color[0]
                        alpha = np.min([np.random.beta(0.01, 0.2), 0.9])
                        ax.fill_between(np.arange(1, num_samples + 1),
                                        np.cumsum(experiment_results) / np.arange(1, num_samples + 1),
                                        np.cumsum(old_result) / np.arange(1, num_samples + 1),
                                        alpha=alpha,
                                        color=tmp_rgb_color)
                    if np.random.uniform(0, 1) > 0.95:
                        tmp_rgb_color = np.array(rand_color.generate(
                            hue=color, luminosity='dark', count=1, format_='Array_rgb')) / 256.
                        tmp_rgb_color = tmp_rgb_color[0]
                        alpha = np.min([np.random.beta(0.1, 0.2), 0.9])
                        linewidth = np.min([np.random.exponential(5.0), 0.9])
                        ax.semilogx(np.arange(1, num_samples + 1),
                                    np.cumsum(experiment_results) / np.arange(1, num_samples + 1),
                                    alpha=alpha,
                                    linewidth=linewidth,
                                    c=tmp_rgb_color)
            old_result = experiment_results[:]

    plt.show()

if __name__ == '__main__':
    # problem2()
    for_fun()
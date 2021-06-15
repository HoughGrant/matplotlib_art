import matplotlib.pyplot as plt
import numpy as np


def run_1():
    colors = [
        # 'viridis',
        # 'plasma',
        # 'inferno',
        # 'magma',
        # 'cividis'
        'Greys',
        # 'Purples',
        # 'Blues',
        # 'PuBu',
        # 'twilight',
        'copper',
        # 'RdYlBu',
        # 'PuOr',
        # 'tab20c'
    ]

    fig = plt.figure(constrained_layout=False, frameon=False)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor((0.7, 0.7, 0.68))
    for i in range(50):
        x = np.random.exponential(i, 50)
        x = np.random.normal(x, 40, 50)
        if np.random.random() > 0.5:
            y = np.random.exponential(2 * i, 50)
        else:
            y = np.random.normal(80, 40, 50)
        ax.hist2d(y, x,
                  # alpha=np.random.uniform(0.2, 0.5),
                  alpha=np.random.beta(2, 5),
                  cmap=np.random.choice(colors), bins=np.random.randint(10, 40))
    plt.show()


if __name__ == '__main__':
    run_1()

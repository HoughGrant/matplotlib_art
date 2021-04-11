from joblib import load
import numpy as np
import matplotlib.pyplot as plt

mlp = load('vector_field_classifier.joblib')


def run_1(cmap: str, background_rgb: tuple, rand_coeff_range: tuple, rand_coeff_samples: int = 1, plot_matrx=True,
          plot_many_series_n=0, plot_series_title='test', antialiased=True, contour_levels=500):
    def make_random_contours():
        output_contours = []
        rand_locs = np.random.choice(np.arange(8), 2, replace=False)
        for x in np.linspace(-5, 5, 100):
            contour_line = []
            for y in np.linspace(-5, 5, 100):
                coeffs = np.array([0, 0, 0, 0, 0, 0, 0, 0]) + np.random.uniform(*rand_coeff_range, rand_coeff_samples)
                coeffs[rand_locs[0]] = x
                coeffs[rand_locs[1]] = y
                z = mlp.predict_proba([coeffs])
                contour_line.append(z[0, 1])
            output_contours.append(contour_line)
        output_contours = np.array(output_contours)
        return output_contours

    if plot_matrx:
        plt.rcParams["figure.facecolor"] = background_rgb
        fig, axes = plt.subplots(3, 3)
        for row in axes:
            for ax in row:
                ax.axis('off')
                ax.contourf(make_random_contours(), cmap=cmap, levels=contour_levels, antialiased=antialiased)
        fig.tight_layout()
        plt.show()

    for i in range(plot_many_series_n):
        fig = plt.figure(frameon=False)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')
        ax.contourf(make_random_contours(), cmap=cmap, levels=contour_levels, antialiased=antialiased)
        fig.savefig(plot_series_title + '_' + str(i))
        plt.close(fig)


if __name__ == '__main__':
    # run_1(cmap='Blues', background_rgb=(123/256, 178/256, 237/256), rand_coeff_range=(-1, 1))
    # run_1(cmap='plasma', background_rgb=(0.3, 0.2, 0.2), rand_coeff_range=(-1, 1))
    # run_1(cmap='Greys', background_rgb=(0.3, 0.2, 0.2), rand_coeff_range=(-1, 1))
    # run_1(cmap='hot', background_rgb=(49/256, 4/256, 6/256), rand_coeff_range=(-1, 1))
    # run_1(cmap='hot', background_rgb=(49/256, 4/256, 6/256), rand_coeff_range=(-1, 1), plot_matrx=True,
    #       contour_levels=500, antialiased=True)
    # run_1(cmap='hot', background_rgb=(49/256, 4/256, 6/256), rand_coeff_range=(-1, 1), plot_matrx=False,
    #       plot_series_title='hot_500levels_antialiasedtrue', plot_many_series_n=5, antialiased=True)
    # run_1(cmap='tab20c', background_rgb=(49/256, 4/256, 6/256), rand_coeff_range=(-1, 1), plot_matrx=False,
    #       plot_series_title='tab20c_antialiased', plot_many_series_n=5, antialiased=True)
    run_1(cmap='Set1', background_rgb=(49/256, 4/256, 6/256), rand_coeff_range=(-1, 1), plot_matrx=False,
          plot_series_title='set1_notantialiased', plot_many_series_n=5, antialiased=False)
    # run_2()

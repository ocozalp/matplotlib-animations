import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# all with zero means
def mv_normal(covariance, x):
    result = 1.0 / np.sqrt(4 * np.pi * np.pi * np.linalg.det(covariance))
    result *= np.exp(-0.5 * x.getT() * np.linalg.inv(covariance) * x)
    return np.asscalar(result)


def metropolis(V, V2, samples, rejects):
    #select a random start point
    if len(samples) == 0:
        x = np.asmatrix([np.random.random(), np.random.random()]).getT()
        samples.append(x)
    else:
        #get last sample
        x = samples[-1]

        result1 = mv_normal(V, x)
        x2 = np.asmatrix(np.random.multivariate_normal([0, 0], V2)).getT()
        result2 = mv_normal(V, x2)
        a = min(1.0, result2/result1)

        if np.random.random() <= a:
            samples.append(x2)
        else:
            rejects.append((x, x2))


#from https://github.com/joferkington/oost_paper_code/blob/master/error_ellipse.py
def draw_ellipse(cov, fig):
    from matplotlib.patches import Ellipse

    def eigen_sorted():
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    ax = fig.gca()

    vals, vecs = eigen_sorted()
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * 2 * np.sqrt(vals)
    ellip = Ellipse(xy=[0, 0], width=width, height=height, angle=theta, fill=False)
    ax.add_artist(ellip)


def generate_random(arg1, V, V2, samples, rejects):
    metropolis(V, V2, samples, rejects)
    rej = draw_rejects(rejects)
    rej.extend(draw_samples(samples))
    return rej


def draw_samples(samples):
    x = list()
    y = list()

    for sample in samples:
        x.append(np.asscalar(sample[0]))
        y.append(np.asscalar(sample[1]))

    return plt.plot(x, y, 'y')


def draw_rejects(rejects):
    results = list()
    for reject in rejects:
        pt1 = reject[0]
        pt2 = reject[1]

        results.extend(plt.plot([np.asscalar(pt1[0]), np.asscalar(pt2[0])],
                                [np.asscalar(pt1[1]), np.asscalar(pt2[1])], 'r'))

    return results


def main():
    fig = plt.figure()
    plt.ylim([-3, 3])
    plt.xlim([-3, 3])

    samples = list()
    rejects = list()

    V = np.asmatrix([[0.5, 0.2], [0.2, 0.5]])
    V2 = np.asmatrix([[0.25, 0], [0, 0.25]])

    draw_ellipse(V, fig)

    line_ani = animation.FuncAnimation(fig, generate_random, 25, fargs=(V, V2, samples, rejects),
                                       interval=250, blit=True)

    line_ani.save('animation.gif', writer='imagemagick', fps=2)
    plt.show()


if __name__ == '__main__':
    main()
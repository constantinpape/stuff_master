import matplotlib.pyplot as plot
import numpy as np

def weighting_function(probs, m):
    x    = np.power( np.divide( probs, 1. - probs ), m )
    ret = np.divide( x, 1. + x )
    return ret


def plot_fixed_prob(p):
    x = np.arange(0., 15., 0.01)
    y = weighting_function(p, x)

    plot.rc('text', usetex = True)
    plot.rc('font', family = 'serif')

    plot.plot(x, y, label = "Probability = " + str(p) )

    plot.title("Weighting with fixed probability")
    plot.xlabel(r'$\mathbf{weight}$')
    plot.ylabel(r'$\mathbf{p^{\star}}$')
    plot.legend()

    plot.show()


def plot_fixed_weight(m):
    x = np.arange(0., 1., 0.01)
    y = weighting_function(x, m)

    plot.rc('text', usetex = True)
    plot.rc('font', family = 'serif')

    plot.plot(x, y, label = "Weight = " + str(m))

    plot.title("Weighting with fixed weight")
    plot.xlabel(r'$\mathbf{p}$')
    plot.ylabel(r'$\mathbf{p^{\star}}$')
    plot.legend()

    plot.show()

if __name__ == '__main__':
    plot_fixed_weight(5)
    #plot_fixed_prob(0.8)

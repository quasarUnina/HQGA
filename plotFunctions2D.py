from mpl_toolkits.mplot3d import Axes3D  

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

import Problems as p
import intervalCode as code


def display(problem):
    X = code.getAllValues(problem.lower_bounds[0], problem.upper_bounds[0], problem.num_bit_code)
    l = []
    for x in X:
        chr_real = [x]
        value = problem.computeFitnessFromReal(chr_real)
        l.append(value)
    Y = np.array(l)
    plt.plot(X, Y, color="black")
    plt.show()

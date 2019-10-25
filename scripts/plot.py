from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
    
def cross_validation_visualization_log(lambds, mse_tr, mse_te):
    """visualization the curves of mse_tr and mse_te."""
    plt.figure(figsize=(20,10))
    plt.semilogx(lambds, mse_tr, marker=".", color='b', label='train error')
    plt.semilogx(lambds, mse_te, marker=".", color='r', label='test error')
    plt.xlabel("lambda")
    plt.ylabel("rmse")
    plt.title("cross validation")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("cross_validation")

def cross_validation_visualization(degree, mse_tr, mse_te):
    """visualization the curves of mse_tr and mse_te."""
    plt.figure(figsize=(20,10))
    plt.plot(degree, mse_tr, marker=".", color='b', label='train error')
    plt.plot(degree, mse_te, marker=".", color='r', label='test error')
    plt.xlabel("degree")
    plt.ylabel("rmse")
    plt.title("cross validation")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("cross_validation")


def plot_3d(degrees, lambdas, rmse):
    fig = plt.figure(num=None, figsize=(15, 10), dpi=80, facecolor='w', edgecolor='k')
    ax = fig.gca(projection='3d')


    X, Y = np.meshgrid(degrees, lambdas)
    Z = rmse.reshape(Y.shape)

    # Plot the surface.
    graph = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)


    ax.set_xlabel('Degree Parameter')
    ax.set_ylabel('Lambda Parameter')
    ax.set_zlabel('RMSE')

    # Customize the z axis.
    #ax.set_zlim(0, 1.00)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(graph, shrink=0.5, aspect=5)

    plt.show()

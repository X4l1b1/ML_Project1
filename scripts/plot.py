# ********************************************************** #
# Project 1 : CS-433 Machine Learning Class                  #
# Various regression models to detect Higgs particles        #
# Authors: Arthur Passuello, Francois Quellec, Julien Muster #
# ********************************************************** #

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import seaborn as sns
    
def cross_validation_visualization_log(lambds, mse_tr, mse_te):
    """visualization the curves of mse_tr and mse_te."""
    plt.figure(figsize=(20,10))
    plt.semilogx(lambds, mse_tr, marker=".", color='b', label='train accuracy')
    plt.semilogx(lambds, mse_te, marker=".", color='r', label='test accuracy')
    plt.xlabel("lambda")
    plt.ylabel("accuracy")
    plt.title("cross validation")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("cross_validation")

def cross_validation_visualization(degree, mse_tr, mse_te):
    """visualization the curves of mse_tr and mse_te."""
    plt.figure(figsize=(20,10))
    plt.plot(degree, mse_tr, marker=".", color='b', label='train accuracy')
    plt.plot(degree, mse_te, marker=".", color='r', label='test accuracy')
    plt.xlabel("degree")
    plt.ylabel("accuracy")
    plt.title("cross validation")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("cross_validation")


def plot_3d(X, Y, Z):
    """ Plot in 3D """
    fig = plt.figure(num=None, figsize=(15, 10), dpi=80, facecolor='w', edgecolor='k')
    ax = fig.gca(projection='3d')

    # Plot the surface.
    graph = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)


    #ax.set_yscale("log")
    ax.set_xlabel('Degree Parameter')
    ax.set_ylabel('Lambda Parameter')
    ax.set_zlabel('Accuracy')


    ax.set_ylim(1e-12, 0.1)
    # Customize the z axis.
    ax.set_zlim(0.65, 0.87)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(graph, shrink=0.5, aspect=5)

    plt.show()

def plot_3_variables(values):
    """ Plot an heatmap of values """
    fig, (ax) = plt.subplots(1, 1, figsize=(15,10))
    values_pivoted = values.pivot("Degree", "Lambda", "Accuracy")
    hm = sns.heatmap(values_pivoted, 
                     ax=ax,           # Axes in which to draw the plot, otherwise use the currently-active Axes.
                     cmap="coolwarm", # Color Map.
                     #square=True,    # If True, set the Axes aspect to “equal” so each cell will be square-shaped.
                     annot=True, 
                     fmt='.2f',       # String formatting code to use when adding annotations.
                     #annot_kws={"size": 14},
                     linewidths=.05)

    fig.subplots_adjust(top=0.93)
    fig.suptitle('Accuracy of Ridge Regression for differents hyper-parameters values', 
                  fontsize=14, 
                  fontweight='bold')
    # fix for mpl bug that cuts off top/bottom of seaborn viz
    b, t = plt.ylim() # discover the values for bottom and top
    b += 0.5 # Add 0.5 to the bottom
    t -= 0.5 # Subtract 0.5 from the top
    plt.ylim(b, t) # update the ylim(bottom, top) values
    plt.show()

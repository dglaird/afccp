# Import libraries
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.patches import Rectangle
import matplotlib.lines as mlines
import numpy as np
from afccp.core.globals import *

# Set matplotlib default font to Times New Roman
import matplotlib as mpl
mpl.rc('font', family='Times New Roman')


def test_graph():
    """
    Simple graph that I can play around with
    :return:
    """
    # Create figure
    figsize = (16, 10)
    fig, ax = plt.subplots(figsize=figsize, facecolor="white", tight_layout=True, dpi=100)

    x = np.random.random(1500)
    y = np.random.random(1500)
    ax.scatter(x,y)
    fig.show()
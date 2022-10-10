import matplotlib.pyplot as plt
import matplotlib as mpl

SMALL_SIZE = 15
MEDIUM_SIZE = 16
BIGGER_SIZE = 16


mpl.rcParams.update(mpl.rcParamsDefault)
plt.rcParams["axes.prop_cycle"]   = mpl.cycler(color=["darkorchid", "darkorange", "lightskyblue", "dodgerblue", "mediumblue"],
                                               linestyle=[':', '-.', '-', '-', '-'])

plt.rcParams['axes.titlesize']   = MEDIUM_SIZE  # fontsize of the axes title
plt.rcParams['axes.labelsize']   = MEDIUM_SIZE  # fontsize of the x and y labels
plt.rcParams['xtick.labelsize']  = SMALL_SIZE  # fontsize of the tick labels
plt.rcParams['ytick.labelsize']  = SMALL_SIZE  # fontsize of the tick labels
plt.rcParams['legend.fontsize']  = MEDIUM_SIZE  # legend fontsize
plt.rcParams['figure.titlesize'] = BIGGER_SIZE  # fontsize of the figure title
# Set the default color cycle
plt.rcParams['lines.linewidth']  = 3

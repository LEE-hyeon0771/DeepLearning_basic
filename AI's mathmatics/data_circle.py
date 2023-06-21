import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

df = pd.read_csv("Prb_data/Prb02data02.csv", header=None)

cmap = LinearSegmentedColormap.from_list("mycmap", ['navy', 'cyan', 'yellow'])

for i in range(3):
    data = df.iloc[:, i].values
    data = data[::-1]

    figure, axes = plt.subplots()

    norm=plt.Normalize(vmin=np.min(data), vmax=np.max(data))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    plt.colorbar(sm)

    for j in range(len(data)-1, -1, -1):
        color = sm.to_rgba(data[j])
        draw_circle = plt.Circle((0.5, 0.5), 0.5+((j+1)*0.02), facecolor=color, linestyle='-', edgecolor='black')
        axes.add_patch(draw_circle)

    axes.autoscale_view()
    axes.get_xaxis().set_visible(False)
    axes.get_yaxis().set_visible(False)

    plt.show()
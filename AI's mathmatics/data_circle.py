'''import matplotlib.pyplot as plt
import pandas as pd

def get_color(val):
    if val == 1:
        return 'blue'
    elif val == 1.5:
        return 'cyan'
    else:  # val == 2
        return 'yellow'


def draw_concentric_circles(data):
    fig, ax = plt.subplots()

    for i in range(len(data)-1, -1, -1):
        val = data[i]
        color = get_color(val)
        # Create circles with larger radius first and gradually decrease the radius as you move inside
        circle = plt.Circle((0, 0), radius=len(data)-i, color=color, fill=False, edgecolor='black', linewidth=12)
        ax.add_patch(circle)

    innermost_color = get_color(data[-1])
    center_circle = plt.Circle((0, 0), radius=0.9, color=innermost_color, edgecolor='black')
    ax.add_patch(center_circle)
    # Set equal aspect ratio and limit x, y axis to show full circle
    ax.set_aspect('equal', 'box')
    ax.set_xlim([-len(data), len(data)])
    ax.set_ylim([-len(data), len(data)])

    # Show the plot
    plt.show()

df = pd.read_csv('Prb_data/Prb02data02.csv', header=None)

data1 = df.iloc[:, 0].tolist()
data2 = df.iloc[:, 1].tolist()
data3 = df.iloc[:, 2].tolist()

print(data1)
print(data2)
print(data3)

draw_concentric_circles(data1)
draw_concentric_circles(data2)
draw_concentric_circles(data3)'''

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

# 색상 매핑을 위한 선형 분할 컬러맵 생성
cmap = LinearSegmentedColormap.from_list("mycmap", ['blue', 'cyan', 'yellow'])

def draw_concentric_circles(data):
    fig, ax = plt.subplots()

    for i in range(len(data)-1, -1, -1):
        val = data[i]
        color = cmap((val - 1) / (2 - 1))  # Normalize val to 0-1 for the color mapping
        # Create circles with larger radius first and gradually decrease the radius as you move inside
        circle = plt.Circle((0, 0), radius=len(data)-i, color=color, fill=False, edgecolor='black', linewidth=12)
        ax.add_patch(circle)

    innermost_color = cmap((data[-1] - 1) / (2 - 1))  # Normalize the innermost data to 0-1 for the color mapping
    center_circle = plt.Circle((0, 0), radius=0.9, color=innermost_color, edgecolor='black')
    ax.add_patch(center_circle)
    # Set equal aspect ratio and limit x, y axis to show full circle
    ax.set_aspect('equal', 'box')
    ax.set_xlim([-len(data), len(data)])
    ax.set_ylim([-len(data), len(data)])

    # Show the plot
    plt.show()

df = pd.read_csv('Prb_data/Prb02data02.csv', header=None)

data1 = df.iloc[:, 0].tolist()
data2 = df.iloc[:, 1].tolist()
data3 = df.iloc[:, 2].tolist()

draw_concentric_circles(data1)
draw_concentric_circles(data2)
draw_concentric_circles(data3)
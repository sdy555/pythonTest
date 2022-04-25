import cv2
import matplotlib.pyplot as plt

import numpy as np

from matplotlib.patches import Ellipse, Circle

from matplotlib.path import Path

# Get an example image

img = cv2.imread('D:/aiImg/yibiao.jpg')

# Create a figure. Equal aspect so circles look circular

fig,ax = plt.subplots(1)

ax.set_aspect('equal')

# Show the image

ax.imshow(img)

ax.set_xlim(0,1600)

ax.set_ylim(0,1200)

# Now, loop through coord arrays, and create a circle at each x,y pair

ellipse = Ellipse((1000, 400), width=400, height=100, edgecolor='white',facecolor='none',linewidth=2)

ax.add_patch(ellipse)

path = ellipse.get_path()

# Show the image

plt.show()
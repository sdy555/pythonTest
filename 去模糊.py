import cv2
from skimage import data,io,filters
import matplotlib.pyplot as plt
img = io.imread('D:/aiImg/588.jpg')
edges1 = filters.gaussian(img,sigma=0.4) #sigma=0.4
edges2 = filters.gaussian(img,sigma=0.1) #sigma=5
plt.figure('gaussian',figsize=(8,8))
plt.subplot(121)
plt.imshow(img)
plt.subplot(122)
plt.imshow(edges2,plt.cm.gray)
plt.show()

io.imshow(edges2)
io.imsave('55.jpg',edges2)



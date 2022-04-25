import cv2
import numpy as np
from skimage import exposure, io

np.set_printoptions(threshold=np.inf)
image = cv2.imread('D:/Users/sdy555/PycharmProjects/pythonProject1/function/duibi2.png')
# image = cv2.imread('D:/aiImg/img/8.jpg')
hue_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
low_range = np.array([160, 103, 100])
high_range = np.array([190, 255, 255])

# low_range = np.array([150, 103, 100])
# high_range = np.array([180, 255, 255])
th = cv2.inRange(hue_image, low_range, high_range)
index1 = th == 255

img = np.zeros(image.shape, np.uint8)
img[:, :] = (255,255,255)
img[index1] = image[index1]#(0,0,255)
cv2.imshow('img', img)
cv2.waitKey(0)

import cv2
import numpy as np

np.set_printoptions(threshold=np.inf)  # 当数组元素比较多的时候，如果输出该数组，那么会出现省略号
image = cv2.imread("0002.png")
image = cv2.resize(image, None, fx=0.5, fy=0.5)

hue_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
low_range = np.array([150, 103, 170])
high_range = np.array([190, 255, 255])
th = cv2.inRange(hue_image, low_range, high_range)
index1 = th == 255

img = np.zeros(image.shape, np.uint8)
img[:, :] = (255, 255, 255)
img[index1] = image[index1]  # (0,0,255)
img0 = image
cv2.imshow('original_img', image)
cv2.imshow('extract_img', img)
cv2.imwrite('original_img.png', image)
cv2.imwrite('extract_img.png', img)


# import cv2
# import numpy as np
#
# image0 = cv2.imread("D:/aiImg/se.png", cv2.IMREAD_COLOR)  # 以BGR色彩读取图片
# B_channel, G_channel, R_channel = cv2.split(image0)
# print(R_channel.shape)
import cv2
from matplotlib import pyplot as plt

# img_bgr = cv2.imread('D:/aiImg/se.png')
# img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
# img_h = img_hsv[..., 0]
# img_s = img_hsv[..., 1]
# img_v = img_hsv[..., 2]
#
# fig = plt.gcf()                      # 分通道显示图片
# fig.set_size_inches(10, 15)
#
# plt.subplot(221)
# plt.imshow(img_hsv)
# plt.axis('off')
# plt.title('HSV')
#
# plt.subplot(222)
# plt.imshow(img_h, cmap='gray')
# plt.axis('off')
# plt.title('H')
#
# plt.subplot(223)
# plt.imshow(img_s, cmap='gray')
# plt.axis('off')
# plt.title('S')
#
# plt.subplot(224)
# plt.imshow(img_v, cmap='gray')
# plt.axis('off')
# plt.title('V')
#
# plt.show()

import cv2
import numpy as np

# set red thresh
# lower_blue=np.array([156,43,46])
# upper_blue=np.array([180,255,255])

lower_blue = np.array([10, 43, 46])
upper_blue = np.array([100, 255, 255])

img = cv2.imread('D:/aiImg/11.jpg')

# get a frame and show

frame = img

cv2.imshow('Capture', img)

# change to hsv model
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

# get mask
mask = cv2.inRange(hsv, lower_blue, upper_blue)
cv2.imshow('Mask', mask)

# detect red
res = cv2.bitwise_and(frame, frame, mask=mask)
cv2.imshow('Result', res)

cv2.waitKey(0)
cv2.destroyAllWindows()
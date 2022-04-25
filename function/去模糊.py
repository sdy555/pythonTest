
# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
# img = cv2.imread('D:/Users/sdy555/PycharmProjects/pythonProject1/function/4.png')
# #cv2.bilateralFilter(src, d, sigmaColor, sigmaSpace)
# #d – Diameter of each pixel neighborhood that is used during filtering.
# # If it is non-positive, it is computed from sigmaSpace
# #9 邻域直径，两个 75 分别是空间高斯函数标准差，灰度值相似性高斯函数标准差
# blur = cv2.bilateralFilter(img,9,75,75)
# plt.subplot(121),plt.imshow(img),plt.title('Original')
# plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(blur),plt.title('Blurred')
# plt.xticks([]), plt.yticks([])
# plt.show()
# !/usr/bin/env python
# _*_ coding:utf-8 _*_
import cv2 as cv
import numpy as np


def custom_blur_demo(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)  # 锐化
    dst = cv.filter2D(image, -1, kernel=kernel)
    cv.imshow("custom_blur_demo", dst)


src = cv.imread("D:/Users/sdy555/PycharmProjects/pythonProject1/function/4.png")
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
cv.imshow("input image", src)
custom_blur_demo(src)
cv.waitKey(0)
cv.destroyAllWindows()
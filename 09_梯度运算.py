import cv2 as cv
import numpy as np

# pie = cv.imread('D:/aiImg/pie.png')
img = cv.imread('D:/aiImg/dige.png')
kernel = np.ones((7, 7), np.uint8)
# dilate = cv.dilate(pie, kernel, iterations=5)
# erosion = cv.erode(pie, kernel, iterations=5)
#
# res = np.hstack((dilate, erosion))
#
# # cv.imshow('res', res)
# # cv.waitKey(0)
# # cv.destroyWindows()
# # 梯度运算  梯度= 膨胀-腐蚀
# gradient = cv.morphologyEx(pie, cv.MORPH_GRADIENT, kernel)
# cv.imshow('gradient', gradient)
# cv.waitKey(0)
# cv.destroyWindows()


# 礼帽与黑帽
# 礼貌= 原始输入-开元算结果
# 黑帽= 闭运算-原始输入
#  礼帽
# tophat = cv.morphologyEx(img, cv.MORPH_TOPHAT, kernel)
# cv.imshow('tophat', tophat)
# cv.waitKey(0)
# cv.destroyWindows()

# 黑帽
blackhat = cv.morphologyEx(img, cv.MORPH_BLACKHAT, kernel)
cv.imshow('blackhat', blackhat)
cv.waitKey(0)
cv.destroyWindows()

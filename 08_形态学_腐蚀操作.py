import cv2 as cv
import numpy as np

# img_dige = cv.imread('D:/aiImg/dige.png')
#
# # cv.imshow('img_dige',img_dige)
# # cv.waitKey(0)
# # cv.destroyAllWindows()
#
# kernel = np.ones((5,5),np.uint8)
# erosion = cv.erode(img_dige, kernel, iterations = 1)
#
# cv.imshow('erosion', erosion)
# cv.waitKey(0)
# cv.destroyAllWindows()
#
# pie = cv.imread('D:/aiImg/pie.png')

# cv.imshow('pie', pie)
# cv.waitKey(0)
# cv.destroyAllWindows()


# kernel = np.ones((30, 30), np.uint8)
# erosion_1 = cv.erode(pie, kernel, iterations=1)
# erosion_2 = cv.erode(pie, kernel, iterations=2)
# erosion_3 = cv.erode(pie, kernel, iterations=3)
# res = np.hstack((erosion_1, erosion_2, erosion_3))
# cv.imshow('res', res)
# cv.waitKey(0)
# cv.destroyAllWindows()


# 膨胀操作
# kernel = np.ones((30, 30), np.uint8)
# dilate_1 = cv.dilate(pie, kernel, iterations=1)
# dilate_2 = cv.dilate(pie, kernel, iterations=2)
# dilate_3 = cv.dilate(pie, kernel, iterations=3)
# res = np.hstack((dilate_1, dilate_2, dilate_3))
# cv.imshow('res', res)
# cv.waitKey(0)
# cv.destroyAllWindows()

#开:先腐蚀，在膨胀
# img = cv.imread('D:/aiImg/dige.png')
#
# kernel = np.ones((5, 5), np.uint8)
# opening = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
#
# cv.imshow('opening',opening)
# cv.waitKey(0)
# cv.destroyAllWindows()


# 闭:先膨胀，在腐蚀
img = cv.imread('D:/aiImg/dige.png')

kernel = np.ones((5,5),np.uint8)
closing = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)

cv.imshow('closing',closing)
cv.waitKey(0)
cv.destroyAllWindows()



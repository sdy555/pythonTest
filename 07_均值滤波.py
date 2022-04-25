import cv2 as cv
import numpy as np

img = cv.imread("D:/aiImg/lenaNoise.png")

#
# cv.imshow('name',img)
# cv.waitKey(0)
# cv.destroyAllWindows()

# 均值滤波
blur = cv.blur(img, (3, 3))

cv.imshow('blur', blur)
cv.waitKey(0)
cv.destroyAllWindows()

# # 方框滤波
# box = cv.boxFilter(img, -1, (3, 3), normalize=True)
#
# cv.imshow('box', box)
# cv.waitKey(0)
# cv.destroyAllWindows()

# 高斯滤波
aussian = cv.GaussianBlur(img, (5, 5), 1)

cv.imshow('aussian', aussian)
cv.waitKey(0)
cv.destroyAllWindows()


# 中值滤波
median = cv.medianBlur(img, 5)

cv.imshow('median', median)
cv.waitKey(0)
cv.destroyAllWindows()


# 展示所有的
res = np.hstack((blur, aussian, median))
print(res)
cv.imshow('median vs average', res)
cv.waitKey(0)
cv.destroyAllWindows()



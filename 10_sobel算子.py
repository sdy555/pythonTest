import cv2 as cv
import numpy as np

img = cv.imread('D:/aiImg/car.png', cv.IMREAD_GRAYSCALE)


def cv_show(name, img):
    cv.imshow(name, img)
    cv.waitKey(0)
    cv.destroyAllWindows()


# sobel算子
# sobelx = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=3)
# sobelx = cv.convertScaleAbs(sobelx)
# sobely = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=3)
# sobely = cv.convertScaleAbs(sobely)
#
# # # 分别计算x，y在求和
# sobelxy = cv.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
# # cv_show('sobelxy', sobelxy)
#
# # 图像梯度算子-Scharr算子
# scharrx = cv.Scharr(img, cv.CV_64F, 1, 0)
# scharrx = cv.convertScaleAbs(scharrx)
# scharry = cv.Scharr(img, cv.CV_64F, 0, 1)
# scharry = cv.convertScaleAbs(scharry)
# scharrxy = cv.addWeighted(scharrx, 0.5, scharry, 0.5, 0)
#
# # laplacian算子
# laplacian = cv.Laplacian(img,cv.CV_64F)
# laplacian = cv.convertScaleAbs(laplacian)
#
# res = np.hstack((sobelxy,scharrxy,laplacian))

v1 = cv.Canny(img, 120, 250)
v2 = cv.Canny(img, 50, 100)

res = np.hstack((v1, v2))

cv_show('res', res)

import cv2 as cv
import numpy as np
img = cv.imread('D:/aiImg/AM.png')


def cv_show(name, img):
    cv.imshow(name, img)
    cv.waitKey(0)
    cv.destroyAllWindows()


#金字塔
# cv_show('as',img)
# print(img.shape)
#
# up = cv.pyrUp(img)
# print(up.shape)
# up2 = cv.pyrUp(up)
# # cv_show('up',up2)
# print(up2.shape)

# cv_show('res', up)

# up = cv.pyrUp(img)
# up_down = cv.pyrDown(up)

# cv_show('up_down',np.hstack((img,up_down)))

# 拉普拉斯金字塔

down = cv.pyrDown(img)
down_up = cv.pyrUp(down)





l_1 = img-down_up
cv_show('l_1',l_1)

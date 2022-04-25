
import cv2 as cv

#img = cv2.imread("D:/aiImg/cat.jpg")

# cv.imshow("opencv-python", src)
# #等待时间，毫秒级，0表示任意键终止
# cv.waitKey(1000)
# cv.destroyAllWindows()

# 定一个函数 图像
# def cv_show(name,img):
#     cv.imshow(name,img)
#     cv.waitKey(0)
#     cv.destroyAllWindows()
#
# cv_show('img', img)

import cv2

# img = cv2.imread("./test.jpg")
img = cv2.imread("D:/aiImg/cat.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, contours, -1, (0, 0, 255), 3)

cv2.imshow("img", img)
cv2.waitKey(0)

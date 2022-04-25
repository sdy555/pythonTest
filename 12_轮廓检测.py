import cv2 as cv

def cv_lunkuojianche(img):
    img = cv.imread(img)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    draw_img = img.copy()
    res = cv.drawContours(draw_img, contours, -1, (0, 0, 255), 2)
# cv_show('res', res)
    cv.imshow('res',res)
    cv.waitKey(0)
    cv.destroyAllWindows()

cv_lunkuojianche('D:/aiImg/contours.png')


# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# ret, binary = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
#
# contours, hierarchy = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
# cv.drawContours(img, contours, -1, (0, 0, 255), 3)

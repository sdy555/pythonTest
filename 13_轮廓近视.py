import cv2 as cv


# img = cv.imread('D:/aiImg/contours2.png')
#
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# ret, thresh = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
# contours,hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
# cnt = contours[0]
#
# def cv_show(name,img):
#     cv.imshow(name,img)
#     cv.waitKey(0)
#     cv.destroyAllWindows()
#
# # draw_img = img.copy()
# # res = cv.drawContours(draw_img, [cnt], -1,(0,0,255), 2)
#
# epsilon = 0.1*cv.arcLength(cnt,True)
# approx = cv.approxPolyDP(cnt,epsilon,True)
# draw_img = img.copy()
# res = cv.drawContours(draw_img, [approx], -1,(0,0,255), 2)
# cv_show('res',res)

# 边界矩形 boundingRect  rectangle
img = cv.imread('D:/aiImg/contours.png')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
contours,hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
cnt = contours[0] #0是变量 ，
x,y,w,h=cv.boundingRect(cnt)
img = cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)



area = cv.contourArea(cnt)
x,y,w,h=cv.boundingRect(cnt)
rect_area = w*h
extent = float(area)/rect_area
cv.imshow('img', img)
cv.waitKey(0)
cv.destroyAllWindows()
print('轮廓面积与边界矩形比',extent)
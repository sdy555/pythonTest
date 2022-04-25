import cv2 as cv
import numpy as np

# img = cv.imread('D:/aiImg/lena.jpg',0)
# template = cv.imread('D:/aiImg/face.jpg',0)
# h,w= template.shape[:2]
# # .TM_SQDIFF:计算平方不同,计算出来的值越小,越相关
# # .TM_CCORR:计算相关性,计算出来的值越大,越相关
# # .TM_CCOEFF:计算相关系数,计算出来的值越大,越相关
# # .TM_SQDIFF_NORMED:计算归一化平方不同,计算出来的值越接近0,越相关
# # .TM_CCORR_NORMED:计算归一化相关性,计算出来的值越接近1,越相关
# # .TM_CCOEFF_NORMED:计算归一化相关系数,计算出来的值越接近1,越相关
# methods=['cv.TM_CCOEFF','cv.TM_CCOEFF_NORMED','cv.TM_CCORR',
#          'cv.TM_CCORR_NOREMD','cv.TM_SQDIFF','cv.TM_SQDIFF_NORMED']
# res = cv.matchTemplate(img,template,cv.TM_SQDIFF)
# min_val, max_val,min_loc,max_loc = cv.minMaxLoc(res)
# print(min_val)


# 模板匹配多个对象

img_rgb = cv.imread('D:/aiImg/mario.jpg')
img_gray = cv.cvtColor(img_rgb,cv.COLOR_BGR2GRAY)
template = cv.imread('D:/aiImg/mario_coin.jpg',0)
h,w = template.shape[:2]

res = cv.matchTemplate(img_gray,template,cv.TM_CCOEFF_NORMED)
threshold  = 0.8
# 取匹配程度大于%80的坐标
loc = np.where(res >= threshold)
for pt in zip(*loc[::-1]):
    bottom_right = (pt[0] + w, pt[1]+h)
    cv.rectangle(img_rgb,pt,bottom_right,(0,0,255),2)

cv.imshow('img_rab',img_rgb)
cv.waitKey(0)


# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
# import math
# img=cv2.imread("D:/aiImg/58.jpg",3)
# #img=cv2.blur(img,(1,1))
# imgray=cv2.Canny(img,600,100,3)#Canny边缘检测，参数可更改
# #cv2.imshow("0",imgray)
# ret,thresh = cv2.threshold(imgray,127,255,cv2.THRESH_BINARY)
# contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)#contours为轮廓集，可以计算轮廓的长度、面积等
# for cnt in contours:
#     if len(cnt)>50:
#         S1=cv2.contourArea(cnt)
#         ell=cv2.fitEllipse(cnt)
#         S2 =math.pi*ell[1][0]*ell[1][1]
#         if (S1/S2)>0.2 :#面积比例，可以更改，根据数据集。。。
#             img = cv2.ellipse(img, ell, (0, 255, 0), 2)
#             print(str(S1) + "    " + str(S2)+"   "+str(ell[0][0])+"   "+str(ell[0][1]))
# cv2.imshow("0",img)
# cv2.waitKey(0)


# import cv2
# import imageio
# import matplotlib.pyplot as plt
#
# from skimage import data, color, img_as_ubyte, io
# from skimage.feature import canny
# from skimage.transform import hough_ellipse
# from skimage.draw import ellipse_perimeter
#
# # Load picture, convert to grayscale and detect edges
# img =cv2.imread('D:/aiImg/yibiao.jpg')
# image_rgb = img[0:220, 160:420]
# image_gray = color.rgb2gray(image_rgb)
# edges = canny(image_gray, sigma=2.0,
#               low_threshold=0.55, high_threshold=0.8)
#
# # Perform a Hough Transform
# # The accuracy corresponds to the bin size of a major axis.
# # The value is chosen in order to get a single high accumulator.
# # The threshold eliminates low accumulators
# result = hough_ellipse(edges, accuracy=20, threshold=250,
#                        min_size=100, max_size=120)
# # result.sort(order='accumulator')
# print(result)
# # Estimated parameters for the ellipse
# # best = list(result[-1])
# # yc, xc, a, b = [int(round(x)) for x in best[1:5]]
# # orientation = best[5]
#
# # Draw the ellipse on the original image
# cy, cx = ellipse_perimeter(r=100,c=100,r_radius=80,c_radius=80)
# image_rgb[cy, cx] = (0, 0, 255)
# # Draw the edge (white) and the resulting ellipse (red)
# edges = color.gray2rgb(img_as_ubyte(edges))
# edges[cy, cx] = (250, 0, 0)
#
# fig2, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(8, 4),
#                                 sharex=True, sharey=True)
#
# ax1.set_title('Original picture')
# ax1.imshow(image_rgb)
#
# ax2.set_title('Edge (white) and result (red)')
# ax2.imshow(edges)
#
# plt.show()



# import cv2
# import matplotlib.pyplot as plt
#
# from skimage import data, color, img_as_ubyte
# from skimage.feature import canny
# from skimage.transform import hough_ellipse
# from skimage.draw import ellipse_perimeter
#
# # Load picture, convert to grayscale and detect edges
# # image_rgb = data.coffee()[0:220, 160:420]
# image_rgb = cv2.imread('D:/aiImg/58.jpg')
# image_gray = color.rgb2gray(image_rgb)
# edges = canny(image_gray, sigma=2.0,
#               low_threshold=0.55, high_threshold=0.8)
#
# # Perform a Hough Transform
# # The accuracy corresponds to the bin size of a major axis.
# # The value is chosen in order to get a single high accumulator.
# # The threshold eliminates low accumulators
# result = hough_ellipse(edges, accuracy=20, threshold=250,
#                        min_size=100, max_size=120)
# result.sort(order='accumulator')
#
# # Estimated parameters for the ellipse
# best = list(result[-1])
# yc, xc, a, b = [int(round(x)) for x in best[1:5]]
# orientation = best[5]
#
# # Draw the ellipse on the original image
# cy, cx = ellipse_perimeter(yc, xc, a, b, orientation)
# image_rgb[cy, cx] = (0, 0, 255)
# # Draw the edge (white) and the resulting ellipse (red)
# edges = color.gray2rgb(img_as_ubyte(edges))
# edges[cy, cx] = (250, 0, 0)
#
# fig2, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(8, 4),
#                                 sharex=True, sharey=True)
#
# ax1.set_title('Original picture')
# ax1.imshow(image_rgb)
#
# ax2.set_title('Edge (white) and result (red)')
# ax2.imshow(edges)
#
# plt.show()
import cv2


import matplotlib.pyplot as plt

from skimage import data, io, color, img_as_ubyte, transform
from skimage.feature import canny
from skimage.transform import hough_ellipse
from skimage.draw import ellipse_perimeter

# Load picture, convert to grayscale and detect edges
# image_rgb = data.coffee()[0:220, 160:420]
# image_rgb = .imread('D:/aiImg/yibiao.jpg')
# image_rgb = cv2.imread('D:/aiImg/5.jpg')
# image_rgb = cv2.imread('D:/aiImg/yibiao.jpg')
#
# # image_rgb = transform.resize(image_rgb1,(80,50))
# image_gray = color.rgb2gray(image_rgb)
# edges = canny(image_gray, sigma=2.0,
#               low_threshold=0.55, high_threshold=0.8)
#
# # Perform a Hough Transform
# # The accuracy corresponds to the bin size of a major axis.
# # The value is chosen in order to get a single high accumulator.
# # The threshold eliminates low accumulators
# result = hough_ellipse(edges, accuracy=20, threshold=250,
#                        min_size=100, max_size=120)
# result.sort(order='accumulator')
# print(result)
# # Estimated parameters for the ellipse
# best = list(result[-1])
# yc, xc, a, b = [int(round(x)) for x in best[1:5]]
# orientation = best[5]
#
# # Draw the ellipse on the original image
# cy, cx = ellipse_perimeter(yc, xc, a, b, orientation)
# image_rgb[cy, cx] = (0, 0, 255)
# # Draw the edge (white) and the resulting ellipse (red)
# edges = color.gray2rgb(img_as_ubyte(edges))
# edges[cy, cx] = (250, 0, 0)
#
# fig2, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(8, 4),
#                                 sharex=True, sharey=True)
#
# ax1.set_title('Original picture')
# ax1.imshow(image_rgb)
#
# ax2.set_title('Edge (white) and result (red)')
# ax2.imshow(edges)
#
# plt.show()
#霍夫圆检测

# import  cv2
#
# #载入并显示图片
# img=cv2.imread("D:/aiImg/yibiao.jpg")
# cv2.imshow("img",img)
# #灰度化
# gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# #输出图像大小，方便根据图像大小调节minRadius和maxRadius
# print(img.shape)
# #霍夫变换圆检测
# circles= cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,1,100,param1=100,param2=30,minRadius=5,maxRadius=300)
# #输出返回值，方便查看类型
# print(circles)
# #输出检测到圆的个数
# print(len(circles[0]))
#
# print("-------------我是条分割线-----------------")
# #根据检测到圆的信息，画出每一个圆
# for circle in circles[0]:
#     #圆的基本信息
#     print(circle[2])
#     #坐标行列
#     x=int(circle[0])
#     y=int(circle[1])
#     #半径
#     r=int(circle[2])
#     #在原图用指定颜色标记出圆的位置
#     img=cv2.circle(img,(x,y),r,(0,0,255),-1)
# #显示新图像
# cv2.imshow("res",img)
#
# #按任意键退出
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# import cv2
# import numpy as np
#
# # read image
# img = cv2.imread('D:/aiImg/zhang.png')
# hh, ww = img.shape[:2]
#
# # convert to grayscale
# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#
# # threshold to binary and invert
# thresh = cv2.threshold(gray, 252, 255, cv2.THRESH_BINARY)[1]
#
# # fit ellipse
# # note: transpose needed to convert y,x points from numpy to x,y for opencv
# points = np.column_stack(np.where(thresh.transpose() > 0))
# hull = cv2.convexHull(points)
# ((centx,centy), (width,height), angle) = cv2.fitEllipse(hull)
# print("center x,y:",centx,centy)
# print("diameters:",width,height)
# print("orientation angle:",angle)
#
# # draw ellipse on input image
# result = img.copy()
# cv2.ellipse(result, (int(centx),int(centy)), (int(width/2),int(height/2)), angle, 0, 360, (0,0,255), 2)
#
# # show results
# cv2.imshow('image', img)
# cv2.imshow('thresh', thresh)
# cv2.imshow('result', result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# save results
# cv2.imwrite('ellipse_shape_fitted.png', result)

# import cv2 as cv
# import numpy as np
#
# def detect_circles_demo(image):
#     dst = cv.pyrMeanShiftFiltering(image, 10, 100)   #边缘保留滤波EPF
#     cimage = cv.cvtColor(dst, cv.COLOR_RGB2GRAY)
#     circles = cv.HoughCircles(cimage, cv.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)
#     circles = np.uint16(np.around(circles)) #把circles包含的圆心和半径的值变成整数
#     for i in circles[0, : ]:
#         cv.circle(image, (i[0], i[1]), i[2], (0, 0, 255), 2)  #画圆
#         cv.circle(image, (i[0], i[1]), 2, (0, 0, 255), 2)  #画圆心
#     cv.imshow("circles", image)
#
# src = cv.imread('D:/aiImg/6.png')
# cv.namedWindow('input_image', cv.WINDOW_NORMAL) #设置为WINDOW_NORMAL可以任意缩放
# cv.imshow('input_image', src)
# detect_circles_demo(src)
# cv.waitKey(0)
# cv.destroyAllWindows()

import os
import cv2
import numpy as np

image = cv2.imread('D:/aiImg/zhang.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.medianBlur(gray, 3)

circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 200, param1=100, param2=30, minRadius=0, maxRadius=0)

for i in circles[0, :]:
    cv2.circle(image, (i[0], i[1]), i[2], (0, 0, 255), 2)  # 画圆
    cv2.circle(image, (i[0], i[1]), 2, (0, 0, 255), 2)  # 画圆心

# cv2.imwrite('circles.jpg', image)

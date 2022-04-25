# import cv2
# import numpy as np
#
# img = cv2.imread('D:/aiImg/xie.jpg')
#
# result3 = img.copy()
#
# img = cv2.GaussianBlur(img,(3,3),0)
# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#
# edges = cv2.Canny(gray,50,150,apertureSize = 3)
# cv2.imwrite("canny.jpg", edges)
#
# src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
# dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])
# m = cv2.getPerspectiveTransform(src, dst)
# result = cv2.warpPerspective(result3, m, (337, 488))
# cv2.imshow("result", result)
# cv2.waitKey(0)

# -*- coding: UTF-8 -*-

import numpy as np
import cv2


## 图片旋转
# def rotate_bound(image, angle):
#     # 获取宽高
#     (h, w) = image.shape[:2]
#     (cX, cY) = (w // 2, h // 2)
#
#     # 提取旋转矩阵 sin cos
#     M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
#     cos = np.abs(M[0, 0])
#     sin = np.abs(M[0, 1])
#
#     # 计算图像的新边界尺寸
#     nW = int((h * sin) + (w * cos))
#     #     nH = int((h * cos) + (w * sin))
#     nH = h
#
#     # 调整旋转矩阵
#     M[0, 2] += (nW / 2) - cX
#     M[1, 2] += (nH / 2) - cY
#
#     return cv2.warpAffine(image, M, (nW, nH), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
#
#
# ## 获取图片旋转角度
# def get_minAreaRect(image):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     gray = cv2.bitwise_not(gray)
#     thresh = cv2.threshold(gray, 0, 255,
#                            cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
#     coords = np.column_stack(np.where(thresh > 0))
#     return cv2.minAreaRect(coords)
#
#
# image_path = "D:/aiImg/xie.jpg"
# image = cv2.imread(image_path)
# angle = get_minAreaRect(image)[-1]
# rotated = rotate_bound(image, angle)
#
# cv2.putText(rotated, "angle: {:.2f} ".format(angle),
#             (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#
# # show the output image
# print("[INFO] angle: {:.3f}".format(angle))
# cv2.imshow("imput", image)
# cv2.imshow("output", rotated)
# cv2.waitKey(0)

# coding=utf-8
# import cv2
# import numpy as np
# import math
#
# def fourier_demo():
#     #1、灰度化读取文件，
#     img = cv2.imread('D:/aiImg/xie.jpg',0)
#
#     #2、图像延扩
#     h, w = img.shape[:2]
#     new_h = cv2.getOptimalDFTSize(h)
#     new_w = cv2.getOptimalDFTSize(w)
#     right = new_w - w
#     bottom = new_h - h
#     nimg = cv2.copyMakeBorder(img, 0, bottom, 0, right, borderType=cv2.BORDER_CONSTANT, value=0)
#     cv2.imshow('new image', nimg)
#
#     #3、执行傅里叶变换，并过得频域图像
#     f = np.fft.fft2(nimg)
#     fshift = np.fft.fftshift(f)
#     magnitude = np.log(np.abs(fshift))
#
#
#     #二值化
#     magnitude_uint = magnitude.astype(np.uint8)
#     ret, thresh = cv2.threshold(magnitude_uint, 11, 255, cv2.THRESH_BINARY)
#     print(ret)
#
#     cv2.imshow('thresh', thresh)
#     print(thresh.dtype)
#     #霍夫直线变换
#     lines = cv2.HoughLinesP(thresh, 2, np.pi/180, 30, minLineLength=40, maxLineGap=100)
#     print(len(lines))
#
#     #创建一个新图像，标注直线
#     lineimg = np.ones(nimg.shape,dtype=np.uint8)
#     lineimg = lineimg * 255
#
#     piThresh = np.pi/180
#     pi2 = np.pi/2
#     print(piThresh)
#
#     for line in lines:
#         x1, y1, x2, y2 = line[0]
#         cv2.line(lineimg, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         if x2 - x1 == 0:
#             continue
#         else:
#             theta = (y2 - y1) / (x2 - x1)
#         if abs(theta) < piThresh or abs(theta - pi2) < piThresh:
#             continue
#         else:
#             print(theta)
#
#     angle = math.atan(theta)
#     print(angle)
#     angle = angle * (180 / np.pi)
#     print(angle)
#     angle = (angle - 90)/(w/h)
#     print(angle)
#
#     center = (w//2, h//2)
#     M = cv2.getRotationMatrix2D(center, angle, 1.0)
#     rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
#     cv2.imshow('line image', lineimg)
#     cv2.imshow('rotated', rotated)
#
# fourier_demo()
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#旋转图像矫正

# import cv2
# import numpy as np
#
# def Img_Outline(input_dir):
#     original_img = cv2.imread(input_dir)
#     gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray_img, (9, 9), 0)                     # 高斯模糊去噪（设定卷积核大小影响效果）
#     _, RedThresh = cv2.threshold(blurred, 165, 255, cv2.THRESH_BINARY)  # 设定阈值165（阈值影响开闭运算效果）
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))          # 定义矩形结构元素
#     closed = cv2.morphologyEx(RedThresh, cv2.MORPH_CLOSE, kernel)       # 闭运算（链接块）
#     opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)           # 开运算（去噪点）
#     return original_img, gray_img, RedThresh, closed, opened
#
#
# def findContours_img(original_img, opened):
#     image, contours, hierarchy = cv2.findContours(opened, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#     c = sorted(contours, key=cv2.contourArea, reverse=True)[1]          # 计算最大轮廓的旋转包围盒
#     rect = cv2.minAreaRect(c)
#     angle = rect[2]
#     print("angle",angle)
#     box = np.int0(cv2.boxPoints(rect))
#     draw_img = cv2.drawContours(original_img.copy(), [box], -1, (0, 0, 255), 3)
#     rows, cols = original_img.shape[:2]
#     M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
#     result_img = cv2.warpAffine(original_img, M, (cols, rows))
#     return result_img,draw_img
#
#
# if __name__ == "__main__":
#     input_dir = ""
#     original_img, gray_img, RedThresh, closed, opened = Img_Outline(input_dir)
#     result_img,draw_img = findContours_img(original_img,opened)
#
#     cv2.imshow("original_img", original_img)
#     cv2.imshow("gray_img", gray_img)
#     cv2.imshow("RedThresh", RedThresh)
#     cv2.imshow("Close", closed)
#     cv2.imshow("Open", opened)
#     cv2.imshow("draw_img", draw_img)
#     cv2.imshow("result_img", result_img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# 基于透视的图像矫正
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# img = cv2.imread('D:/aiImg/xie2.jpg')
# H_rows, W_cols= img.shape[:2]
# print(H_rows, W_cols)
#
# # 原图中书本的四个角点(左上、右上、左下、右下),与变换后矩阵位置
# pts1 = np.float32([[161, 80], [449, 12], [1, 430], [480, 394]])
# pts2 = np.float32([[0, 0],[W_cols,0],[0, H_rows],[H_rows,W_cols],])
#
# # 生成透视变换矩阵；进行透视变换
# M = cv2.getPerspectiveTransform(pts1, pts2)
# dst = cv2.warpPerspective(img, M, (500,470))
#
# """
# 注释代码同效
# # img[:, :, ::-1]是将BGR转化为RGB
# # plt.subplot(121), plt.imshow(img[:, :, ::-1]), plt.title('input')
# # plt.subplot(122), plt.imshow(dst[:, :, ::-1]), plt.title('output')
# # plt.show
# """
#
# cv2.imshow("original_img",img)
# cv2.imshow("result",dst)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# 自动获取图像顶点变换
# from imutils.perspective import four_point_transform
# import imutils
# import cv2
#
# def Get_Outline(input_dir):
#     image = cv2.imread(input_dir)
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (5,5),0)
#     edged = cv2.Canny(blurred,75,200)
#     return image,gray,edged
#
# def Get_cnt(edged):
#     cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
#     cnts = cnts[0] if  imutils.is_cv2()  else   cnts[1]
#     docCnt =None
#
#     if len(cnts) > 0:
#         cnts =sorted(cnts,key=cv2.contourArea,reverse=True)
#         for c in cnts:
#             peri = cv2.arcLength(c,True)                   # 轮廓按大小降序排序
#             approx = cv2.approxPolyDP(c,0.02 * peri,True)  # 获取近似的轮廓
#             if len(approx) ==4:                            # 近似轮廓有四个顶点
#                 docCnt = approx
#                 break
#     return docCnt
#
# if __name__=="__main__":
#     input_dir = "D:/aiImg/xie3.jpg"
#     image,gray,edged = Get_Outline(input_dir)
#     docCnt = Get_cnt(edged)
#     result_img = four_point_transform(image, docCnt.reshape(4,2)) # 对原始图像进行四点透视变换
#     cv2.imshow("original", image)
#     cv2.imshow("gray", gray)
#     cv2.imshow("edged", edged)
#     cv2.imshow("result_img", result_img)
#
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2


def correctImage(path):
    img = cv2.imread('D:/aiImg/xie2.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    """将图片转化成二值图像"""
    ret, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    """
    *** cv2.finContours 函数能求取图像的边界，
    返回的第一个值是用list存储的边界的数组每个边界又有多个点来组成
    *** cv2.RETR_EXTERNAL 参数的含义是仅求取外围的边界
    """
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    """
    *** 下面的循环遍历了边界，找到含有最多点的那个边界的下标
    *** 这是我观察发现的(没有科学的论证)，有其中一个边界的点是最多的，基本能通过这个边界画出物体的轮廓
    """
    m = 0;
    index = 0
    for i in range(len(contours)):
        if len(contours[i]) > m:
            m = len(contours[i])
            index = i
    """
    *** cv2.minAreaRect 函数能根据点集求最小的外接矩形
    (rect, rect[0] = 矩形中心, rect[1] = 矩形长和宽, rect[3] = 矩形的角度)
    角度的大小是[0,90], 表示的是x轴逆时针旋转到与rect的一边重合的角度，重合的那条边是举行的宽。
    """
    rect = cv2.minAreaRect(contours[index])
    """
    *** cv2.drawContours(  #函数能将边界画到图像里
        binary,            #第一个参数是目标图像
        contours=contours, #边界集
        contourIdx=-1,     #选择的边界的序号，-1表示选择集合内的所有的边界
        thickness=-1,      #表示边界内的图形的填充方式，-1表示filled方式，也就是填满颜色
        color=(255) )      #表示填充的颜色，255表示填充白色
    """
    dst = np.zeros((img.shape))
    cv2.drawContours(binary, contours=contours, contourIdx=-1, color=(255), thickness=-1)
    """
    cv2.copyTo(src,mask) 根据mask(掩码)来将img填充到dst
    """
    dst = cv2.copyTo(img, binary)
    """
    *** cv2.getRotationMatrix2D( #根据输入的参数，返回旋转矩阵(因为对图像的旋转操作是采用的矩阵相乘)
        centry, #旋转中心
        angel,  #旋转角度
        scale   #缩放
    )
    *** cv2.warpAffine(对图像进行旋转和缩放
        1st, #输入图像
        2sc, #旋转矩阵
        3rd, #输出图像大小
        4th, #输出
        5th, #flag
        6th  #填充颜色
    )
    """
    angel = abs(rect[2])
    if angel > 60:
        angel = 90 - angel
    mat = cv2.getRotationMatrix2D(rect[0], angel, 1)
    rotimg = np.zeros((img.shape))
    rotimg = cv2.warpAffine(dst, mat, (img.shape[1], img.shape[0]), rotimg, 1, 0)

    im1 = img
    im2 = binary
    im3 = dst
    im4 = rotimg
    plt.subplot(2, 2, 1), plt.title('source')
    plt.imshow(im1, )
    plt.subplot(2, 2, 2), plt.title('contours')
    plt.imshow(im2, 'gray')
    plt.subplot(2, 2, 3), plt.title('cppyTo')
    plt.imshow(im3)
    plt.subplot(2, 2, 4), plt.title('final')
    plt.imshow(im4)
    plt.show()


correctImage('')



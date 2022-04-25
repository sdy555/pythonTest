# import cv2
#
# img_path = "D:/img_imread/000002.jpg"
# #读取文件
# mat_img = cv2.imread(img_path)
# mat_img2 = cv2.imread(img_path,cv2.CV_8UC1)
#
# #自适应分割
# dst = cv2.adaptiveThreshold(mat_img2,210,cv2.BORDER_REPLICATE,cv2.THRESH_BINARY_INV,3,10)
# #提取轮廓
# contours,heridency = cv2.findContours(dst,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
# #标记轮廓
# cv2.drawContours(mat_img,contours,-1,(255,0,255),3)
#
# #计算轮廓面积
# area = 0
# for i in contours:
#     area += cv2.contourArea(i)
# print(area)
#
# #图像show
# cv2.imshow("window1",mat_img)
# cv2.waitKey(0)
#
# import numpy as np
# import cv2
# import os
# image = 'D:/aiImg/img/1.png'
# # savefile = './'
# # image = os.listdir(image_file)
# # save_image = os.path.join(savefile, image)
#
# #设定颜色HSV范围，假定为红色
# redLower = np.array([156, 43, 46])
# redUpper = np.array([179, 255, 255])
#
# #读取图像
# img = cv2.imread(image)
#
# #将图像转化为HSV格式
# hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#
# #去除颜色范围外的其余颜色
# mask = cv2.inRange(hsv, redLower, redUpper)
#
# # 二值化操作
# ret, binary = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
#
# #膨胀操作，因为是对线条进行提取定位，所以腐蚀可能会造成更大间隔的断点，将线条切断，因此仅做膨胀操作
# kernel = np.ones((5, 5), np.uint8)
# dilation = cv2.dilate(binary, kernel, iterations=1)
#
# #获取图像轮廓坐标，其中contours为坐标值，此处只检测外形轮廓_,
# contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# print(contours)
# print(hierarchy)
# if len(contours) > 0:
#     #cv2.boundingRect()返回轮廓矩阵的坐标值，四个值为x, y, w, h， 其中x, y为左上角坐标，w,h为矩阵的宽和高
#     boxes = [cv2.boundingRect(c) for c in contours]
#     for box in boxes:
#         x, y, w, h = box
#         #绘制矩形框对轮廓进行定位
#         cv2.rectangle(img, (x, y), (x+w, y+h), (153, 153, 0), 2)
#         #将绘制的图像保存并展示
# 	# cv2.imwrite(save_image, img)
#
#     cv2.namedWindow('image', cv2.WINDOW_NORMAL)
#     cv2.resizeWindow('image', 600, 800)  # 改变窗口大小
#     cv2.imshow('image', img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# import xlwt
#
# img = cv2.imread('D:/img_imread/000002.jpg')
# hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# low_hsv = np.array([0, 0, 221])
# high_hsv = np.array([180, 30, 255])
# mask = cv2.inRange(hsv, lowerb=low_hsv, upperb=high_hsv)
#
# print(len(mask))
# print(len(mask[0]))
#
# list_y = []
# list_x = []
#
# for i in range(len(mask)):
#     # print(mask[i])
#     xmax = []
#     for j in range(len(mask[i])):
#         if mask[i][j] == 0:
#             # print(mask[i][j],j,i)
#             list_x.append(j)
#             list_y.append(len(mask)-i)



# wb.save('1111.xls')


# import cv2
# import numpy as np
#
#
# # 读取图片
# def print_u1(img1):
#     img1=cv2.imread(img1,1)
#     def ReadImg():
#         img = img1
#         # cv2.imshow('src', img)
#         return img
#
#
#     # 高斯滤波
#     def GausBlur(src):
#         dst = cv2.GaussianBlur(src, (5, 5), 1.5)
#         # cv2.imshow('GausBlur', dst)
#         return dst
#
#
#     # 灰度处理
#     def Gray_img(src):
#         gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
#         # cv2.imshow('gray', gray)
#         return gray
#
#
#     # 二值化
#     def threshold_img(src):
#         ret, binary = cv2.threshold(src, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)
#         print("threshold value %s" % ret)
#         # cv2.imshow('threshold', binary)
#         return binary
#
#
#     # 开运算操作
#     def open_mor(src):
#         kernel = np.ones((5, 5), np.uint8)
#         opening = cv2.morphologyEx(src, cv2.MORPH_OPEN, kernel, iterations=3)  # iterations进行3次操作
#         # cv2.imshow('open', opening)
#         return opening
#
#
#     # 轮廓拟合
#     def draw_shape(open_img, gray_img):
#         contours, hierarchy = cv2.findContours(open_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#         cnt = contours[0]  # 得到第一个的轮廓
#
#         rect = cv2.minAreaRect(cnt)
#         box = cv2.boxPoints(rect)
#         box = np.int0(box)
#         cv2.drawContours(src, [box], 0, (0, 0, 255), 3)  # 画矩形框
#
#         # 图像轮廓及中心点坐标
#         M = cv2.moments(cnt)  # 计算第一条轮廓的各阶矩,字典形式
#         center_x = int(M['m10'] / M['m00'])
#         center_y = int(M['m01'] / M['m00'])
#         print('center_x:', center_x)
#         print('center_y:', center_y)
#         cv2.circle(src, (center_x, center_y), 7, 128, -1)  # 绘制中心点
#         str1 = '(' + str(center_x) + ',' + str(center_y) + ')'  # 把坐标转化为字符串
#         cv2.putText(src, str1, (center_x - 50, center_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2,
#                     cv2.LINE_AA)  # 绘制坐标点位
#
#         # cv2.imshow('show', src)
#
#
#     src = ReadImg()
#     gaus_img = GausBlur(src)
#     gray_img = Gray_img(gaus_img)
#     thres_img = threshold_img(gray_img)
#     open_img = open_mor(thres_img)
#     draw_shape(open_img, src)
# print_u1('D:/aiImg/img/1.png')
# cv2.waitKey(0)
import json
import os

import cv2

# img = cv2.imread('D:/aiImg/img/1.png')
# imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# ret,dst = cv2.threshold(imgray,127,255,0)
# cnts, hierarchy =cv2.findContours(dst,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
# rows,cols = img.shape[:2]
# [vx,vy,x,y] = cv2.fitLine(cnts[0], cv2.DIST_L2,0,0.01,0.01)
# lefty = int((-x*vy/vx) + y)
# righty = int(((cols-x)*vy/vx)+y)
# cv2.line(img,(cols-1,righty),(0,lefty),(0,255,0),2)
# print (img.shape[:2])
# cv2.imshow('image1',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# from skimage import exposure, io
#
#
# def repair(img_imread,img_output): #修复
#     data_base_dir = img_imread  # 输入文件夹的路径
#     outfile_dir = img_output  # 截取红章文件夹放置路径
#     processed_number = 0  # 统计处理图片的数量
#     for file in os.listdir(data_base_dir):  # 遍历目标文件夹图片
#         read_img_name = data_base_dir + '//' + file.strip()  # 取图片完整路径
#         img = cv2.imread(read_img_name)
#         while(1):
#
#             img_rescale = exposure.equalize_hist(img)
#             out_img_name = outfile_dir + '//' + file.strip()
#             io.imsave(out_img_name,img_rescale)
#             processed_number += 1
#             print("修复的照片数量为:", processed_number)
#             break
# repair('D:/img_output','D:/img_output4')

# 打印线条坐标
from skimage import exposure, io

from watchdog.events import *
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.exposure import match_histograms
# def print_u1(img):
#     imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     ret, dst = cv2.threshold(imgray, 127, 255, 0)
#     cnts, hierarchy = cv2.findContours(dst, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     rows, cols = img.shape[:2]
#     [vx, vy, x, y] = cv2.fitLine(cnts[0], cv2.DIST_L2, 0, 0.01, 0.01)
#     lefty = int((-x * vy / vx) + y)
#     righty = int(((cols - x) * vy / vx) + y)
#     cv2.line(img, (cols - 1, righty), (0, lefty), (0, 255, 0), 2)
#     str_img=str((img.shape[:2]))
#     print(str_img)
#     data2 = json.loads(str_img)
#     print('线条坐标：', data2)
#
# img = cv2.imread('D:/img_output2/1.png')
# print_u1(img)
def pretreatment(image_reade, img_output):
    data_base_dir = image_reade  # 输入文件夹的路径
    outfile_dir = img_output  # 输出文件夹的路径
    reference = cv2.imread('D:/aiImg/6.png')  # 目标图像
    processed_number = 0  # 统计处理图片的数量

    for file in os.listdir(data_base_dir):  # 遍历目标文件夹图片
        read_img_name = data_base_dir + '//' + file.strip()  # 取图片完整路径
        image = cv2.imread(read_img_name)  # 读入图片

        while (1):
            matched = match_histograms(image, reference, channel_axis=-1)
            # cv2.imshow("demo", matched)
            # k = cv2.waitKey(1)
            # if k == 13:  # 按回车键确认处理、保存图片到输出文件夹和读取下一张图片
            processed_number += 1
            out_img_name = outfile_dir + '//' + file.strip()
            cv2.imwrite(out_img_name, matched)
            print("预处理的照片数为", processed_number)
            break


def Eliminate_red_chapter(img_imread, img_output):
    data_base_dir = img_imread  # 输入文件夹的路径
    outfile_dir = img_output  # 截取红章文件夹放置路径
    # quhongzhang_dir = 'D:/aiImg/quchuzhang' #去除红章文件夹放置路径
    processed_number = 0  # 统计处理图片的数量
    for file in os.listdir(data_base_dir):  # 遍历目标文件夹图片
        read_img_name = data_base_dir + '//' + file.strip()  # 取图片完整路径
        image = cv2.imread(read_img_name)  # 读入图片
        while (1):
            red_channel = image[:, :, 2]
            green_channel = image[:, :, 1]
            blue_channel = image[:, :, 0]
            processed_number += 1
            # 或者使用cv2自带的函数,但是耗时比较多
            # channels = cv2.split(image)

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

            ret, red_binary = cv2.threshold(image, 160, 255, cv2.THRESH_BINARY)

            # 合并各个通道的函数
            merge = cv2.merge([blue_channel, green_channel, red_channel])
            out_img_name = outfile_dir + '//' + file.strip()
            cv2.imwrite(out_img_name, red_channel)
            # cv2.imshow("binary", binary)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            break

Eliminate_red_chapter('D:/img_output', 'D:/img_output')# 消除红章
# pretreatment('D:/img_imread', 'D:/img_output1')
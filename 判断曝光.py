import numpy

import cv2 as cv

import numpy as np

import copy

import random

import math

from PIL import Image

from PIL import ImageStat #就靠他了
# data_base_dir = "D:\opencvImg\input"  # 输入文件夹的路径
# outfile_dir = "D:\opencvImg\output"  # 输出文件夹的路径
# processed_number = 0  # 统计处理图片的数量
# print
# "press enter to make sure your operation and process the next picture"
#
# for file in os.listdir(data_base_dir):  # 遍历目标文件夹图片
#     read_img_name = data_base_dir + '//' + file.strip()  # 取图片完整路径
#     image = cv2.imread(read_img_name)  # 读入图片

# img = cv.imread('D:/aiImg/mario.jpg',0)
# cv.imshow('img', img)
# cv.waitKey(0)
#
# def brightness2( img ):
#     im = Image.open(img).convert('L')
#     stat = ImageStat.Stat(im)
#     return stat.rms[0]
#
# print(brightness2('D:/aiImg/mario.jpg'))

# import cv2
# imagePath ='D:/aiImg/mario.jpg'
# image = cv2.imread(imagePath)
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# result = cv2.Laplacian(gray, cv2.CV_64F).var()
# cv2.imshow('image',result)
# cv2.waitKey(0)
# print('ai_hellohello.jpg blur:',result )
# import cv2
#
# imagePath ='D:/aiImg/mario.jpg'
# image = cv2.imread(imagePath)
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# result = cv2.Laplacian(gray, cv2.CV_64F).var()
#
# a=8000.5
#     if a<result:
#         print('flase')
#     else:
#         print('true')







import cv2
def panduan(img_url,exposed):
    imagePath =img_url
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    result = cv2.Laplacian(gray, cv2.CV_64F).var()
    if exposed < result:
        return False
        # print('Ture')
    else:
        return True
        # print('Flase')
# print(result)
# print(a)




print(panduan('D:/aiImg/mario.jpg',8000))



# print(type(a))
# print(type(result))
# print('y1 blur:',result)


# imagePath ='D:/aiImg/chessboard.jpg'
# image = cv2.imread(imagePath)
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# print('y2 blur:',cv2.Laplacian(gray, cv2.CV_64F).var())




import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.ndimage as ndi

import plotly
import plotly.express as px
# from skimage import (
#     color, data, measure
# )
# image = data.astronaut()
# image = color.rgb2gray(image)
# blurred_images = [ndi.uniform_filter(image, size=k) for k in range(2, 32, 2)]
# img_stack = np.stack(blurred_images)
#
# fig = px.imshow(
#     img_stack,
#     animation_frame=0,
#     binary_string=True,
#     labels={'animation_frame': 'blur strength ~'}
# )
# plotly.io.show(fig)




# import cv2
# import numpy as np
#
#
# image = cv2.imread('D:/aiImg/yibiao.jpg')
#
# img2gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
# imageVar = cv2.Laplacian(img2gray, cv2.CV_64F).var()
#
# # res = np.hstack((image,imageVar))
#
#     # return imageVar
#
#
# cv2.imshow('hah', image)
# cv2.imshow('two',imageVar )
# cv2.waitKey(0)
# cv2.destroyAllWindows()

import cv2
import numpy as np
import math

# brenner梯度函数计算
def brenner(img):
    '''
    :param img:narray 二维灰度图像
    :return: float 图像越清晰越大
    '''
    shape = np.shape(img)
    out = 0
    for x in range(0, shape[0]-2):
        for y in range(0, shape[1]):
            out+=(int(img[x+2,y])-int(img[x,y]))**2
    return out

#Laplacian梯度函数计算
def Laplacian(img):
    '''
    :param img:narray 二维灰度图像
    :return: float 图像越清晰越大
    '''
    return cv2.Laplacian(img,cv2.CV_64F).var()

#SMD梯度函数计算
def SMD(img):
    '''
    :param img:narray 二维灰度图像
    :return: float 图像越清晰越大
    '''
    shape = np.shape(img)
    out = 0
    for x in range(1, shape[0]-1):
        for y in range(0, shape[1]):
            out+=math.fabs(int(img[x,y])-int(img[x,y-1]))
            out+=math.fabs(int(img[x,y]-int(img[x+1,y])))
    return out

#SMD2梯度函数计算
def SMD2(img):
    '''
    :param img:narray 二维灰度图像
    :return: float 图像越清晰越大
    '''
    shape = np.shape(img)
    out = 0
    for x in range(0, shape[0]-1):
        for y in range(0, shape[1]-1):
            out+=math.fabs(int(img[x,y])-int(img[x+1,y]))*math.fabs(int(img[x,y]-int(img[x,y+1])))
    return out

#方差函数计算
def variance(img):
    '''
    :param img:narray 二维灰度图像
    :return: float 图像越清晰越大
    '''
    out = 0
    u = np.mean(img)
    shape = np.shape(img)
    for x in range(0,shape[0]):
        for y in range(0,shape[1]):
            out+=(img[x,y]-u)**2
    return out

#energy函数计算
def energy(img):
    '''
    :param img:narray 二维灰度图像
    :return: float 图像越清晰越大
    '''
    shape = np.shape(img)
    out = 0
    for x in range(0, shape[0]-1):
        for y in range(0, shape[1]-1):
            out+=((int(img[x+1,y])-int(img[x,y]))**2)*((int(img[x,y+1]-int(img[x,y])))**2)
    return out

#Vollath函数计算
def Vollath(img):
    '''
    :param img:narray 二维灰度图像
    :return: float 图像越清晰越大
    '''
    shape = np.shape(img)
    u = np.mean(img)
    out = -shape[0]*shape[1]*(u**2)
    for x in range(0, shape[0]-1):
        for y in range(0, shape[1]):
            out+=int(img[x,y])*int(img[x+1,y])
    return out

#entropy函数计算
def entropy(img):
    '''
    :param img:narray 二维灰度图像
    :return: float 图像越清晰越大
    '''
    out = 0
    count = np.shape(img)[0]*np.shape(img)[1]
    p = np.bincount(np.array(img).flatten())
    for i in range(0, len(p)):
        if p[i]!=0:
            out-=p[i]*math.log(p[i]/count)/count
    return out

def main(img1, img2):
    print('Brenner',brenner(img1),brenner(img2))
    print('Laplacian',Laplacian(img1),Laplacian(img2))
    print('SMD',SMD(img1), SMD(img2))
    print('SMD2',SMD2(img1), SMD2(img2))
    print('Variance',variance(img1),variance(img2))
    print('Energy',energy(img1),energy(img2))
    print('Vollath',Vollath(img1),Vollath(img2))
    print('Entropy',entropy(img1),entropy(img2))

if __name__ == '__main__':
    #读入原始图像
    img1 = cv2.imread('D:/aiImg/output/666.jpg')
    img2 = cv2.imread('D:/aiImg/output/922.jpg')
    #灰度化处理
    img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    main(img1,img2)


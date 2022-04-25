#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:HP
@file:判断曝光度1.py
@time:2022/03/24
"""
import numpy as np
import cv2
def compute(img, min_percentile, max_percentile):
    """计算分位点，目的是去掉图1的直方图两头的异常情况"""
    max_percentile_pixel = np.percentile(img, max_percentile)
    min_percentile_pixel = np.percentile(img, min_percentile)
    return max_percentile_pixel, min_percentile_pixel
def aug(src,number):
    """图像亮度增强"""
    print('原图亮度：', get_lightness(src))
    print('给定值:',number)
    if get_lightness(src) < number:
        # return "图片曝光度足够，无需增强"
        print("Ture")
        print("图片曝光度足够，返回到指定文件夹")
    else:
        # return False
        print("Flase")
        print("图片曝光曝光度大于给定值，图片已做曝光调整并以输出到指定文件夹")
    # 先计算分位点，去掉像素值中少数异常值，这个分位点可以自己配置。
    # 比如1中直方图的红色在0到255上都有值，但是实际上像素值主要在0到20内。
    max_percentile_pixel, min_percentile_pixel = compute(src, 1, 99)
    # 去掉分位值区间之外的值
    src[src >= max_percentile_pixel] = max_percentile_pixel
    src[src <= min_percentile_pixel] = min_percentile_pixel
    # 将分位值区间拉伸到0到255，这里取了255*0.1与255*0.9是因为可能会出现像素值溢出的情况，所以最好不要设置为0到255。
    out = np.zeros(src.shape, src.dtype)
    cv2.normalize(src, out, number * 0.1, number * 0.9, cv2.NORM_MINMAX)
    print('原图调整后亮度：', get_lightness(out))
    return out

def get_lightness(src):
    # 计算亮度
    hsv_image = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    lightness = hsv_image[:, :, 2].mean()
    return lightness

img= cv2.imread("D:/AiImg/922.jpg")
img = aug(img,180)
cv2.imwrite('D:/aiImg/output/666.jpg', img) # 输出到指定文件
#




# def get_lightness_last(hhh):
#     # 计算亮度
#     hhh = cv2.imread(hhh)
#     hsv_image = cv2.cvtColor(hhh, cv2.COLOR_BGR2HSV)
#     lightness1 = hsv_image[:, :, 2].mean()
#     return lightness1
#
# print('原图调整后亮度：',get_lightness_last(src1))




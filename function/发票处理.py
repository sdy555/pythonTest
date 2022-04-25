#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:HP
@file:流程.py
@time:2022/04/19
"""
import cv2
from skimage import exposure, io
from watchdog.events import *
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.exposure import match_histograms



def pretreatment(image_reade, img_output):
    data_base_dir = image_reade  # 输入文件夹的路径
    outfile_dir = img_output  # 输出文件夹的路径
    reference = cv2.imread('D:/cv2/in/1.png')  # 目标图像
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



def Eliminate_red_chapter(img_imread, img_output):  # 消除红章
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

            ret, red_binary = cv2.threshold(red_channel, 160, 255, cv2.THRESH_BINARY)

            # 合并各个通道的函数
            merge = cv2.merge([blue_channel, green_channel, red_channel])

            out_img_name = outfile_dir + '//' + file.strip()
            cv2.imwrite(out_img_name, red_channel)

            print("消除红章的图片数量为:", processed_number)
            print_width_height(red_channel)
            # print(red_channel)
            # cv2.imshow("binary", binary)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            break


def get_table_structure(img_imread, img_output):  # 获取表结构
    data_base_dir = img_imread  # 输入文件夹的路径
    outfile_dir = img_output  # 截取红章文件夹放置路径
    processed_number = 0  # 统计处理图片的数量
    for file in os.listdir(data_base_dir):  # 遍历目标文件夹图片
        read_img_name = data_base_dir + '//' + file.strip()  # 取图片完整路径
        while (1):
            src_img = cv2.imread(read_img_name)
            src_img0 = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
            src_img0 = cv2.GaussianBlur(src_img0, (3, 3), 0)
            src_img1 = cv2.bitwise_not(src_img0)
            AdaptiveThreshold = cv2.adaptiveThreshold(src_img1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                                      15, -2)

            horizontal = AdaptiveThreshold.copy()
            vertical = AdaptiveThreshold.copy()
            scale = 20

            horizontalSize = int(horizontal.shape[1] / scale)
            horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontalSize, 1))
            horizontal = cv2.erode(horizontal, horizontalStructure)
            horizontal = cv2.dilate(horizontal, horizontalStructure)
            # cv2.imshow("horizontal", horizontal)
            # cv2.waitKey(0)

            verticalsize = int(vertical.shape[1] / scale)
            verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))
            vertical = cv2.erode(vertical, verticalStructure, (-1, -1))
            vertical = cv2.dilate(vertical, verticalStructure, (-1, -1))
            # cv2.imshow("verticalsize", vertical)
            # cv2.waitKey(0)
            processed_number += 1
            mask = horizontal + vertical
            print("获取表结构的图片数量为:", processed_number)
            print_width_height(mask)
            ret, dst = cv2.threshold(mask, 127, 255, 0)
            cnts, hierarchy = cv2.findContours(dst, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            rows, cols = mask.shape[:2]
            [vx, vy, x, y] = cv2.fitLine(cnts[0], cv2.DIST_L2, 0, 0.01, 0.01)
            lefty = int((-x * vy / vx) + y)
            righty = int(((cols - x) * vy / vx) + y)
            cv2.line(mask, (cols - 1, righty), (0, lefty), (0, 255, 0), 2)
            print("x,y坐标为", mask.shape[:2])
            out_img_name = outfile_dir + '//' + file.strip()
            cv2.imwrite(out_img_name, mask)
            # print_width_height(mask)

            cv2.waitKey(0)
            cv2.destroyAllWindows()
            break


def extract_red_chapter(img_imread, img_output):
    np.set_printoptions(threshold=np.inf)
    data_base_dir = img_imread  # 输入文件夹的路径
    outfile_dir = img_output  # 截取红章文件夹放置路径
    processed_number = 0  # 统计处理图片的数量
    for file in os.listdir(data_base_dir):  # 遍历目标文件夹图片
        read_img_name = data_base_dir + '//' + file.strip()  # 取图片完整路径
        while (1):
            image = cv2.imread(read_img_name)
            hue_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            low_range = np.array([150, 103, 170])
            high_range = np.array([190, 255, 255])
            th = cv2.inRange(hue_image, low_range, high_range)
            index1 = th == 255
            processed_number += 1
            out_img_name = outfile_dir + '//' + file.strip()
            img = np.zeros(image.shape, np.uint8)
            img[:, :] = (255, 255, 255)
            img[index1] = image[index1]  # (0,0,255)

            cv2.imwrite(out_img_name, img)

            cv2.waitKey(0)
            print("提取红章的图片数量为:", processed_number)
            print_width_height(img)
            print_xy(image)
            break


def tiqu(img_imread, img_output):
    np.set_printoptions(threshold=np.inf)
    data_base_dir = img_imread  # 输入文件夹的路径
    outfile_dir = img_output  # 截取红章文件夹放置路径
    processed_number = 0  # 统计处理图片的数量
    for file in os.listdir(data_base_dir):  # 遍历目标文件夹图片
        read_img_name = data_base_dir + '//' + file.strip()  # 取图片完整路径
        while (1):
            image = cv2.imread(read_img_name)
            hue_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            low_range = np.array([150, 103, 170])
            high_range = np.array([190, 255, 255])
            th = cv2.inRange(hue_image, low_range, high_range)
            index1 = th == 255
            processed_number += 1
            out_img_name = outfile_dir + '//' + file.strip()
            img = np.zeros(image.shape, np.uint8)
            img[:, :] = (255, 255, 255)
            img[index1] = image[index1]  # (0,0,255)

            cv2.imwrite(out_img_name, img)
            # cv2.waitKey(0)
            # print("输出的照片数量为:", processed_number)
            print_width_height(img)  # 打印宽高
            print_xy(image)  # 打印坐标
            break


def repair(img_imread, img_output):  # 修复
    data_base_dir = img_imread  # 输入文件夹的路径
    outfile_dir = img_output  # 截取红章文件夹放置路径
    processed_number = 0  # 统计处理图片的数量
    for file in os.listdir(data_base_dir):  # 遍历目标文件夹图片
        read_img_name = data_base_dir + '//' + file.strip()  # 取图片完整路径
        img = cv2.imread(read_img_name)
        while (1):
            img_rescale = exposure.equalize_hist(img)
            out_img_name = outfile_dir + '//' + file.strip()
            io.imsave(out_img_name, img_rescale)
            processed_number += 1
            print("修复的照片数量为:", processed_number)
            break


def print_width_height(mask):
    # img = cv2.imread(mask)
    img_rgb = cv2.GaussianBlur(mask, (5, 5), 0)
    canny_img = cv2.Canny(img_rgb, 1, 10)
    H, W = canny_img.shape
    print("图高为:", H, "图宽为:", W)


# 打印坐标
def print_xy(img):
    def extract_red(img):
        ''''使用inRange方法，拼接mask0,mask1'''

        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        rows, cols, channels = img.shape
        # 区间1
        lower_red = np.array([0, 43, 46])
        upper_red = np.array([10, 255, 255])
        mask0 = cv2.inRange(img_hsv, lower_red, upper_red)
        # 区间2
        lower_red = np.array([156, 43, 46])
        upper_red = np.array([180, 255, 255])
        mask1 = cv2.inRange(img_hsv, lower_red, upper_red)
        # 拼接两个区间
        mask = mask0 + mask1
        return mask

    mask = extract_red(img)
    mask_img = cv2.add(img, np.zeros(np.shape(img), dtype=np.uint8), mask=mask)

    # cv2.HoughCircles 寻找出圆，匹配出图章的位置
    binaryImg = cv2.Canny(mask_img, 50, 200)  # 二值化，canny检测
    contours, hierarchy = cv2.findContours(binaryImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    circles = cv2.HoughCircles(binaryImg, cv2.HOUGH_GRADIENT, 1, 40,
                               param1=50, param2=30, minRadius=20, maxRadius=60)
    #
    circles = np.uint16(np.around(circles))

    # findContours(image, mode, method[, contours[, hierarchy[, offset]]]) -> image, contours, hierarchy

    '''
    tuple cir_point:圆心坐标
    int radius:半径
    list points:点集
    float dp: 误差
    int samplingtime:采样次数

    '''
    import random

    def circle_check(cir_point, radius, points, dp=4, samplingtime=30):
        # 根据点到圆心的距离等于半径判断点集是否在圆上
        # 多次抽样出5个点
        # 判断多次抽样的结果是否满足条件
        count = 0
        points = list(points)
        for s in range(samplingtime):
            # 从点集points 中采样一次
            points_samp = random.sample(points, 5)
            # 判断点到圆心的距离是否等于半径
            points_samp = np.array(points_samp[0])
            dist = np.linalg.norm(points_samp - cir_point)
            if dist == radius or abs(dist - radius) <= dp:
                continue
            else:
                count += 1
        if count < 3:
            return True
        else:
            return False

    def circle_map(contours, circles):
        is_stramp = [0] * len(contours)
        circle_point = []
        for cir in circles[0, :]:
            # 获取圆心和半径
            cir_point = np.array((cir[0], cir[1]))
            radius = cir[2]

            # 遍历每一个点集
            for cidx, cont in enumerate(contours):
                # 当轮廓点数少于10 的时候，默认其不是公章轮廓
                if len(cont) < 10:
                    continue
                # 匹配出公章轮廓，并对应出圆心坐标
                stampcheck = circle_check(cir_point, radius, cont, dp=6, samplingtime=40)
                # 如果满足点在圆心上，就将圆心,半径和对应的点记录
                if stampcheck:
                    circle_point.append((cont))
                    is_stramp[cidx] = 1

        return circle_point, is_stramp

    circle_point, is_stramp = circle_map(contours, circles)

    print('x轴,y轴：', circle_point[0])


pretreatment('d:/cv2/fp', 'd:/cv2/outing') #预处理
Eliminate_red_chapter('d:/cv2/outing', 'd:/cv2/Eliminate_red_chapter')  # 消除红章
get_table_structure('d:/cv2/outing', 'd:/cv2/get_table_structure')  # 获取表结构
# tiqu('d:/cv2/outing', 'd:/cv2/extract_red_chapter')  # 提取红章
repair('d:/cv2/Eliminate_red_chapter', 'd:/cv2/repair')  # 修复
extract_red_chapter('d:/cv2/outing', 'd:/cv2/extract_red_chapter')  # 提取红章并打印出坐标位置

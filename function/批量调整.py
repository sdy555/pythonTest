import os
import cv2
import numpy as np



# data_base_dir = "D:\cv2\img"  # 输入文件夹的路径（英文路径）
# outfile_dir = "D:\cv2\img2"  # 输出文件夹的路径
from skimage import io
from skimage.exposure import match_histograms

data_base_dir = "D:/aiImg/input"  # 输入文件夹的路径
outfile_dir = "D:/aiImg/output"  # 输出文件夹的路径
reference = cv2.imread('D:/aiImg/yibiao.jpg')# 目标图像
processed_number = 0  # 统计处理图片的数量


for file in os.listdir(data_base_dir):  # 遍历目标文件夹图片
    read_img_name = data_base_dir + '//' + file.strip()  # 取图片完整路径
    image = cv2.imread(read_img_name)  # 读入图片

    while (1):
        matched = match_histograms(image, reference, channel_axis=-1)
        cv2.imshow("demo", matched)
        k = cv2.waitKey(1)
        if k == 13:  # 按回车键确认处理、保存图片到输出文件夹和读取下一张图片
            processed_number += 1
            out_img_name = outfile_dir + '//' + file.strip()
            cv2.imwrite(out_img_name, matched)
            print("已处理的照片数为",processed_number)
            print("按enter键以确保您的操作并处理下一张图片")
            break




import cv2
import numpy as np
from skimage.exposure import match_histograms
from watchdog.events import *
reference = cv2.imread('D:/aiImg/img/1.png')  # 目标图像


def piliangzhifanghu(image_reade,img_output):
    data_base_dir = image_reade  # 输入文件夹的路径
    outfile_dir = img_output  # 输出文件夹的路径
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
            print("已处理的照片数为",processed_number)
            print("按enter键以确保您的操作并处理下一张图片")
            break

def quhongzhang(img_output1):
    processed_number = 0  # 统计处理图片的数量
    piliangzhifanghu('D:/aiImg/input','D:/aiImg/quchuzhang')


    for file in os.listdir('D:/aiImg/quchuzhang'):
        data_base_dir = 'D:/aiImg/quchuzhang'
        read_img_name = data_base_dir + '//' + file.strip()  # 取图片完整路径
        image = cv2.imread(read_img_name)  # 读入图片
        B_channel, G_channel, R_channel = cv2.split(image)
        # 多传入一个参数cv2.THRESH_OTSU，并且把阈值thresh设为0，算法会找到最优阈值
        thresh, ret = cv2.threshold(R_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # 实测调整为95%效果好一些
        filter_condition = int(thresh * 0.95)

        _, red_thresh = cv2.threshold(R_channel, filter_condition, 255, cv2.THRESH_BINARY)


        # 把图片转回 3 通道
        result_img = np.expand_dims(red_thresh, axis=2)
        result_img = np.concatenate((result_img, result_img, result_img), axis=-1)

        out_img_name1 = 'D:/aiImg/quchuzhang' + '//' + file.strip()
        cv2.imwrite(out_img_name1, result_img)
        print("已处理的照片数为", processed_number)
        break



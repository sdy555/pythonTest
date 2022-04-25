
import cv2
import imageio
import numpy as np
from skimage import exposure
from skimage.exposure import match_histograms
from watchdog.events import *
data_base_dir = 'D:/aiImg/input' # 输入文件夹的路径
outfile_dir = 'D:/aiImg/output' # 截取后红章文件夹放置路径
quhongzhang_dir = 'D:/aiImg/quchuzhang' #去除后红章文件夹放置路径
xiufu_dir = 'D:/aiImg/xiufu'
processed_number = 0  # 统计处理图片的数量





for file in os.listdir(data_base_dir):  # 遍历目标文件夹图片
    read_img_name = data_base_dir + '//' + file.strip()  # 取图片完整路径
    image = cv2.imread(read_img_name)  # 读入图片
    image1 = cv2.imread(read_img_name,cv2.IMREAD_COLOR)  # 读入图片
    reference = cv2.imread('D:/aiImg/img/1.png')  # 目标图像
    while (1):
        # 直方图均衡
        matched = match_histograms(image1, reference, channel_axis=-1)

        #截取红章
        np.set_printoptions(threshold=np.inf)
        hue_image = cv2.cvtColor(matched, cv2.COLOR_BGR2HSV)
        low_range = np.array([150, 103, 100])
        high_range = np.array([180, 255, 255])
        th = cv2.inRange(hue_image, low_range, high_range)
        index1 = th == 255

        img = np.zeros(matched.shape, np.uint8)
        img[:, :] = (255, 255, 255)
        img[index1] = matched[index1]  # (0,0,255)
        processed_number += 1
        out_img_name = outfile_dir + '//' + file.strip()
        cv2.imwrite(out_img_name, img)




        #去除红章
        B_channel, G_channel, R_channel = cv2.split(matched)
        _, RedThresh = cv2.threshold(R_channel, 160, 255, cv2.THRESH_BINARY)
        out_img_name1 = quhongzhang_dir + '//' + file.strip()
        cv2.imwrite(out_img_name1, R_channel)

        # 修复
        img_rescale = exposure.equalize_hist(RedThresh)
        out_img_name1 = xiufu_dir + '//' + file.strip()
        imageio.imsave(out_img_name1, img_rescale)
        print("已处理的照片数为", processed_number)


        break
import cv2
import numpy as np
from skimage import io
from skimage.exposure import exposure, match_histograms
from watchdog.events import *
import json

def pretreatment(image_reade, img_output):
    data_base_dir = image_reade  # 输入文件夹的路径
    # outfile_dir = img_output  # 输出文件夹的路径
    outfile_dir1 = 'D:\img_output'  # 输出红章去取后的文件夹的路径
    outfile_dir2 = 'D:\img_output1'  # 输出红章截取后的文件夹的路径
    outfile_dir3 = 'D:\img_output2'  # 输出线条的文件夹的路径
    reference = cv2.imread('D:/cv2/in/1.png')  # 目标图像
    processed_number = 0  # 统计处理图片的数量

    for file in os.listdir(data_base_dir):  # 遍历目标文件夹图片
        read_img_name = data_base_dir + '//' + file.strip()  # 取图片完整路径
        image = cv2.imread(read_img_name)  # 读入图片

        while (1):
            # 预处理
            matched = match_histograms(image, reference, channel_axis=-1)
            # processed_number += 1
            # out_img_name = outfile_dir + '//' + file.strip()
            # cv2.imwrite(out_img_name, matched)

            #去除红章
            red_channel = matched[:, :, 2]
            green_channel = matched[:, :, 1]
            blue_channel = matched[:, :, 0]
            out_img_name = outfile_dir1 + '//' + file.strip()
            cv2.imwrite(out_img_name, red_channel)
            a, x, y= numb(matched)
            # 提取红章
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
            out_img_name2 = outfile_dir2 + '//' + file.strip()
            cv2.imwrite(out_img_name2, img)

            # 获取线条
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
            # print("获取表结构的图片数量为:", processed_number)
            ret, dst = cv2.threshold(mask, 127, 255, 0)
            cnts, hierarchy = cv2.findContours(dst, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            rows, cols = mask.shape[:2]
            [vx, vy, x, y] = cv2.fitLine(cnts[0], cv2.DIST_L2, 0, 0.01, 0.01)
            lefty = int((-x * vy / vx) + y)
            righty = int(((cols - x) * vy / vx) + y)
            cv2.line(mask, (cols - 1, righty), (0, lefty), (0, 255, 0), 2)
            # print("x,y坐标为", mask.shape[:2])
            out_img_name3 = outfile_dir3 + '//' + file.strip()
            cv2.imwrite(out_img_name3, mask)

            #红章个数和坐标
            def numb(img):
                # 图片简单处理
                # img = cv2.imread(img)  # 读取图片
                GrayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 灰度化
                GrayImage = cv2.medianBlur(GrayImage, 5)  # 中值模糊

                # 阈值处理，输入图片默认为单通道灰度图片
                ret, th1 = cv2.threshold(GrayImage, 127, 255, cv2.THRESH_BINARY)  # 固定阈值二值化
                # threshold为固定阈值二值化
                # 第二参数为阈值
                # 第三参数为当像素值超过了阈值（或者小于阈值，根据type来决定），所赋予的值（一般情况下，都是256色，所以默认最大为255）
                # thresh_binary是基于直方图的二值化操作类型，配合threshold一起使用。此外还有cv2.THRESH_BINARY； cv2.THRESH_BINARY_INV； cv2.THRESH_TRUNC； cv2.THRESH_TOZERO；cv2.THRESH_TOZERO_INV
                th2 = cv2.adaptiveThreshold(GrayImage, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 5)
                # adaptiveThreshold自适应阈值二值化，自适应阈值二值化函数根据图片一小块区域的值来计算对应区域的阈值，从而得到也许更为合适的图片。
                # 第二参数为当像素值超过了阈值（或者小于阈值，根据type来决定），所赋予的值（一般情况下，都是256色，所以默认最大为255）
                # 第三参数为阈值计算方法，类型有cv2.ADAPTIVE_THRESH_MEAN_C，cv2.ADAPTIVE_THRESH_GAUSSIAN_C
                # 第四参数是基于直方图的二值化操作类型，配合threshold一起使用。此外还有cv2.THRESH_BINARY； cv2.THRESH_BINARY_INV； cv2.THRESH_TRUNC； cv2.THRESH_TOZERO；cv2.THRESH_TOZERO_INV
                # 第五参数是图片中分块的大小
                # 第六参数是阈值计算方法中的常数项
                th3 = cv2.adaptiveThreshold(GrayImage, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 5)
                # 同上
                kernel = np.ones((5, 5), np.uint8)  # 创建全一矩阵，数值类型设置为uint8
                erosion = cv2.erode(th2, kernel, iterations=1)  # 腐蚀处理
                dilation = cv2.dilate(erosion, kernel, iterations=1)  # 膨胀处理

                imgray = cv2.Canny(erosion, 30, 100)  # Canny算子边缘检测

                circles = cv2.HoughCircles(imgray, cv2.HOUGH_GRADIENT,
                                           1.5,
                                           500,
                                           param1=90,
                                           param2=20,
                                           minRadius=30,
                                           maxRadius=50)  # 霍夫圆变换
                # 第3参数默认为1
                # 第4参数表示圆心与圆心之间的距离（太大的话，会很多圆被认为是一个圆）
                # 第5参数默认为100
                # 第6参数根据圆大小设置(圆越小设置越小，检测的圆越多，但检测大圆会有噪点)
                # 第7圆最小半径
                # 第8圆最大半径
                circles = np.uint16(np.around(circles))
                # np.uint16数组转换为16位，0-65535
                # np.around返回四舍五入后的值

                P = circles[0]  # 去掉circles数组一层外括号
                for i in P:
                    # 画出外圆
                    cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)  # 第二参数（）内是圆心坐标，第三参数是半径，第四参数（）内是颜色，第五参数是线条粗细
                    # 画出圆心
                    cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)
                # print("圆的个数是：")
                a = len(P)
                for i in P:
                    r = int(i[2])
                    x = int(i[0])
                    y = int(i[1])

                return a, x, y




            Results = [{
                'Position':[x,y],
                'Position1':[x,y]

            }]
            data = [{'RecognizeQuantity': processed_number,
                     'ImageName': file.strip(),
                     'Type': 'vat_common_invoice',
                     'SealsQuantity':a,
                     'Data':Results
                     }]
            json_string = json.dumps(data)
            print(json_string)
            break
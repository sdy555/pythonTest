# 引用模块
import PIL.ImageShow
import cv2
import matplotlib.pyplot as plt
import numpy as np


def pinghuachuli(img_url):
    """
    :param img_url:图片地址&变量
    :return:返回所有平滑处理后的图片
    """
    img_url = cv2.imread(img_url)
    edian = cv2.medianBlur(img_url, 5)  # 中值滤波
    aussian = cv2.GaussianBlur(img_url, (5, 5), 1)  # 高斯滤波
    box = cv2.boxFilter(img_url, -1, (3, 3), normalize=True)  # 方框滤波
    box1 = cv2.boxFilter(img_url, -1, (3, 3), normalize=False)  # 方框滤波
    blur = cv2.blur(img_url, (4, 4))  # 均值滤波

    res = np.hstack((edian, aussian, box, box1, blur))
    print(res)
    cv2.imshow('median vs averager', res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



pinghuachuli('D:/aiImg/contours2.png')

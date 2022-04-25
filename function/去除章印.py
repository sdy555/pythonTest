# #去除印章
import cv2
import numpy as np
import matplotlib.pyplot as plt


image0=cv2.imread("D:/aiImg\img/8.jpg",cv2.IMREAD_COLOR)   # 以BGR色彩读取图片
# image0=cv2.imread("D:/aiImg/img/1.png",cv2.IMREAD_COLOR)
# image0=cv2.imread("D:/aiImg/input/000005.jpg",cv2.IMREAD_COLOR)
image = cv2.resize(image0,None,fx=0.5,fy=0.5,
                   interpolation=cv2.INTER_CUBIC)  # 缩小图片0.5倍（图片太大了）
cols,rows,_=image.shape                            # 获取图片高宽
B_channel,G_channel,R_channel=cv2.split(image)     # 注意cv2.split()返回通道顺序

cv2.imshow('Blue channel',B_channel)
cv2.imshow('Green channel',G_channel)
cv2.imshow('Red channel',R_channel)

pixelSequence=R_channel.reshape([rows*cols,])     # 红色通道的histgram 变换成一维向量
numberBins=256                                    # 统计直方图的组数
plt.figure()                                      # 计算直方图
manager = plt.get_current_fig_manager()
histogram,bins,patch=plt.hist(pixelSequence,
                              numberBins,
                              facecolor='black',
                              histtype='bar')     # facecolor设置为黑色
#设置坐标范围
y_maxValue=np.max(histogram)
plt.axis([0,255,0,y_maxValue])
#设置坐标轴
plt.xlabel("gray Level",fontsize=20)
plt.ylabel('number of pixels',fontsize=20)
plt.title("Histgram of red channel", fontsize=25)
plt.xticks(range(0,255,10))
#显示直方图
plt.pause(0.05)
plt.savefig("histgram.png",dpi=260,bbox_inches="tight")
plt.show()


#红色通道阈值(调节好函数阈值为160时效果最好，太大一片白，太小干扰点太多)
_,RedThresh = cv2.threshold(R_channel,160,255,cv2.THRESH_BINARY)

#膨胀操作（可以省略）
element = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
erode = cv2.erode(RedThresh, element)

#显示效果
cv2.imshow('original color image',image)
cv2.imshow("RedThresh",RedThresh)
cv2.imshow("erode",erode)
cv2.waitKey(0)
cv2.destroyAllWindows()
# 保存图像
# cv2.imwrite('scale_image.jpg',image)
# cv2.imwrite('RedThresh.jpg',RedThresh)
# cv2.imwrite("erode.jpg",erode)


# import cv2
#
# image = cv2.imread("D:/Users/sdy555/PycharmProjects/pythonProject1/function/duibi2.png", cv2.IMREAD_COLOR)
# red_channel = image[:, :, 2]
# green_channel = image[:, :, 1]
# blue_channel = image[:, :, 0]
#
# # 或者使用cv2自带的函数,但是耗时比较多
# # channels = cv2.split(image)
#
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
#
# ret, red_binary = cv2.threshold(red_channel, 200, 255, cv2.THRESH_BINARY)
#
# # 合并各个通道的函数
# merge = cv2.merge([blue_channel, green_channel, red_channel])
#
# cv2.imshow("red_binary", red_binary)
# cv2.imshow("red_channel", red_channel)
# cv2.imshow('yuantu',image)
# cv2.imshow('1',merge)
# # cv2.imwrite('./2.jpg',red_channel)
# # cv2.imshow("binary", binary)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#
# -*- encoding: utf-8 -*-
import cv2
import numpy as np


# class SealRemove(object):
#     """
#     印章处理类
#     """
#
#     def remove_red_seal(self, image):
#         """
#         去除红色印章
#         """
#
#         # 获得红色通道
#         blue_c, green_c, red_c = cv2.split(image)
#
#         # 多传入一个参数cv2.THRESH_OTSU，并且把阈值thresh设为0，算法会找到最优阈值
#         thresh, ret = cv2.threshold(red_c, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#         # 实测调整为95%效果好一些
#         filter_condition = int(thresh * 0.95)
#
#         _, red_thresh = cv2.threshold(red_c, filter_condition, 255, cv2.THRESH_BINARY)
#
#         # 把图片转回 3 通道
#         result_img = np.expand_dims(red_thresh, axis=2)
#         result_img = np.concatenate((result_img, result_img, result_img), axis=-1)
#
#         return result_img
#
#
# if __name__ == '__main__':
#     image = 'D:/aiImg/input/122.jpg'
#     img = cv2.imread(image)
#     seal_rm = SealRemove()
#     rm_img = seal_rm.remove_red_seal(img)
#     # cv2.imwrite("D:/test/result.png", rm_img
#     cv2.imshow('rm_img',rm_img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
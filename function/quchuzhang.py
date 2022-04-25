import sys

import cv2
import numpy as np

# image0 = cv2.imread("D:/Users/sdy555/PycharmProjects/pythonProject1/function/duibi2.png", cv2.IMREAD_COLOR)  # 以BGR色彩读取图片
# image0 = cv2.imread("D:/aiImg/8.jpg", cv2.IMREAD_COLOR)  # 以BGR色彩读取图片
# # image = cv2.resize(image0, None, fx=0.5, fy=0.5,
# #                    interpolation=cv2.INTER_CUBIC)  # 缩小图片0.5倍（图片太大了）
# # cols, rows, _ = image.shape  # 获取图片高宽
# B_channel, G_channel, R_channel = cv2.split(image0)  # 注意cv2.split()返回通道顺序
# _, binary = cv2.threshold(R_channel, 145, 255, cv2.THRESH_BINARY)
# ret, red_binary = cv2.threshold(R_channel, 160, 255, cv2.THRESH_BINARY)
# cv2.imshow('Red channel', R_channel)
# cv2.imshow('Red red_binary', red_binary)
# # cv2.imwrite('./3.jpg',red_binary)
# cv2.imshow('original color image', image0)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# import cv2
#
# image = cv2.imread("D:/aiImg/8.jpg.png", cv2.IMREAD_COLOR)
# red_channel = image[:, :, 2]
# green_channel = image[:, :, 1]
# blue_channel = image[:, :, 0]
#
# # 或者使用cv2自带的函数,但是耗时比较多
# # channels = cv2.split(image)
#
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
#
# ret, red_binary = cv2.threshold(red_channel, 160, 255, cv2.THRESH_BINARY)
#
# # 合并各个通道的函数
# merge = cv2.merge([blue_channel, green_channel, red_channel])
#
# cv2.imshow("red_binary", red_binary)
# cv2.imshow("red_channel", red_channel)
# cv2.imshow("binary", binary)
# cv2.waitKey(0)


# import cv2
# import numpy as np
#
# img = cv2.imread('D:/aiImg/8.jpg')
# # 转换为灰度图
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow('img', img)
# cv2.imshow('gray', img_gray), cv2.waitKey(0)
# flags = [i for i in dir(cv2) if i.startswith('COLOR_')]
# print(flags)
# capture = cv2.VideoCapture(0)
#
# # 蓝色的范围，不同光照条件下不一样，可灵活调整
# lower_blue = np.array([100, 110, 110])
# upper_blue = np.array([130, 255, 255])
#
# while (True):
#     # 1.捕获视频中的一帧
#     ret, frame = capture.read()
#
#     # 2.从BGR转换到HSV
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#
#     # 3.inRange()：介于lower/upper之间的为白色，其余黑色
#     mask = cv2.inRange(hsv, lower_blue, upper_blue)
#
#     # 4.只保留原图中的蓝色部分
#     res = cv2.bitwise_and(frame, frame, mask=mask)
#
#     cv2.imshow('frame', frame)
#     cv2.imshow('mask', mask)
#     cv2.imshow('res', res)
#
#     if cv2.waitKey(1) == ord('q'):
#         break
#
# blue = np.uint8([[[255, 0, 0]]])
# hsv_blue = cv2.cvtColor(blue, cv2.COLOR_BGR2HSV)
# print(hsv_blue)  # [[[120 255 255]]]

# import cv2
# import numpy as np
#
#
# def draw_rectangle_r(event, x, y, flags, param):
#     global ix, iy
#
#     if event==cv2.EVENT_LBUTTONDOWN:
#         ix, iy = x, y
#         print("point1:=", x, y)
#         # i+=1
#
#     elif event==cv2.EVENT_LBUTTONUP:
#         print("point2:=", x, y)
#         print("width=",x-ix)
#         print("height=", y - iy)
#
#         roi_area = img[iy:y,ix:x]
#
#         hsv_image = cv2.cvtColor(roi_area, cv2.COLOR_BGR2HSV)
#         high_range = np.array([180, 255, 255])
#         low_range = np.array([156, 43, 46])
#         high_range1 = np.array([10, 255, 255])
#         low_range1 = np.array([0, 43, 46])
#         th = cv2.inRange(hsv_image, low_range, high_range)
#         th1 = cv2.inRange(hsv_image, low_range1, high_range1)
#
#         dst = cv2.inpaint(roi_area, th + th1, 3, cv2.INPAINT_TELEA)
#         # cv2.imshow('dst', dst)
#
#         img[iy:y, ix:x]= dst
#         # cv2.imshow("image", img)
#         cv2.imwrite("new_img.jpg", img)
#         # while(1):
#         cv2.imshow("image",img)
#         if cv2.waitKey(0) & 0xFF == ord('s'):
#             cv2.imwrite("new_img.jpg", img)
#             print("图像已经保存")
#
#
# if __name__ == '__main__':
#
#     img = cv2.imread("D:/aiImg/8.jpg")  #加载图片
#
#     cv2.namedWindow('image')
#     cv2.setMouseCallback('image', draw_rectangle_r)
#
#     cv2.imshow('image', img)
#     # if cv2.waitKey(20) & 0xFF == ord('y'):
#     #     cv2.imshow('image',)
#     #     break
#     # if cv2.waitKey(20) & 0xFF == ord('q'):
#     #     break
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()



# import cv2
# from matplotlib import pyplot as plt
#
# img =cv2.imread("D:/aiImg/input/000001.jpg")
# gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# ret,bin_img = cv2.threshold(gray_img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# # RETR_EXTERNAL:仅外圈轮廓
# contours, hierarchy = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
# cv2.drawContours(img,contours,-1,(0,255,0),5)
# plt.imshow(img[...,::-1])
# plt.axis('off')
#
# print(hierarchy)
# '''
# [[[ 1 -1 -1 -1]
#   [ 2  0 -1 -1]
#   [-1  1 -1 -1]]]
# '''
# #
#

# import cv2
# import numpy as np
#
# img=cv2.imread('D:/aiImg/input/000001.jpg')
#
# def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
#     if event == cv2.EVENT_LBUTTONDOWN:
#         xy = "%d,%d" % (x, y)
#         cv2.circle(img, (x, y), 1, (255, 0, 0), thickness = -1)
#         cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
#                     1.0, (0,0,0), thickness = 1)
#         cv2.imshow("image", img)
# cv2.namedWindow("image")
# cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
# while(1):
#     cv2.imshow("image", img)
#     if cv2.waitKey(0)&0xFF==27:
#         break
# cv2.destroyAllWindows()

# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# import xlwt
#
# img = cv2.imread('D:/aiImg/output/000001.jpg')
# hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# low_hsv = np.array([0, 0, 221])
# high_hsv = np.array([180, 30, 255])
# mask = cv2.inRange(hsv, lowerb=low_hsv, upperb=high_hsv)
#
# print(len(mask))
# print(len(mask[0]))

# from PIL import Image
#
# img = Image.open(f'D:/aiImg/output/000001.jpg')
#
# width = img.size[0]  # 宽
# height = img.size[1] # 高
#
# print(width)
# print(height)
# print(img.format)  # 图片后缀




# from collections import deque
#
# import cv2
# import imutils
#
# # 遍历查看所有颜色空间
# flags = [i for i in dir(cv2) if i.startswith('COLOR_')]
# print(flags)
# print(len(flags))
#
# # 对于 HSV，色调范围为 [0,179]，饱和度范围为 [0,255]，值范围为 [0,255]。不同的软件使用不同的尺度。因此，如果您将 OpenCV 值与它们进行比较，则需要对这些范围进行归一化。
# # 对象跟踪，可以使用HSV来提取彩色对象。这是最简单的对象追踪，找到物体的质心，就可以绘制轨迹来追踪物体；
# import cv2
# import numpy as np
#
# origin = cv2.imread('D:/aiImg/input/000001.jpg')
# frame = origin.copy()
# origin1 = origin.copy()
# cv2.imshow("origin", frame)  # 转换BGR为HSV
# hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
# cv2.imshow("hsv", hsv)
#
# # 定义绿色的HSV空间值
# # lower_blue = np.array([29, 86, 6])
# # upper_blue = np.array([64, 255, 255])
#
# # 定义红色的HSV空间值
# lower_red = np.array([-30, 43, 46])
# upper_red = np.array([30, 255, 255])
#
# # 阈值化图像，只获取红色
# mask = cv2.inRange(hsv, lower_red, upper_red)
# cv2.imshow("mask hsv", mask)
# mask = cv2.dilate(mask, None, iterations=3)
# cv2.imshow("mask dilate", mask)
# mask = cv2.erode(mask, None, iterations=2)
# cv2.imshow("mask erode", mask)
#
# # 寻找mask中的轮廓，并初始化球当前的中心（x,y）
# cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
#                         cv2.CHAIN_APPROX_SIMPLE)
# cnts = imutils.grab_contours(cnts)
# print('cnts: ',cnts)
# center = None
#
# pts = deque(maxlen=500)
# # 发现至少一个轮廓继续处理
# if len(cnts) > 0:
#     for c in cnts:
#         # 计算最小外接圆，质心
#         ((x, y), radius) = cv2.minEnclosingCircle(c)
#         M = cv2.moments(c)
#         center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
#         pts.appendleft(center)
#
#         # 绘制轮廓
#         cv2.drawContours(frame, [c], -1, (0, 255, 0), 2)
#
#         # 在帧上绘制红色球的外接圆及中心
#         # cv2.circle(frame, (int(x), int(y)), int(radius),
#         #            (0, 255, 0), -1)
#         # cv2.circle(frame, (int(x), int(y)), 2,
#         #            (255, 255, 0), 2)
# cv2.imshow("Res", frame)
# cv2.waitKey(0)
#
# # 如何找到HSV空间值呢？
# # 比如要寻找绿色，可以找到BGR(0,255,0) ,转换为HSV，然后取[ H-10,S,V]为下限 -- [ H+10,S,V]为上限；
# green = np.uint8([[[0, 255, 0]]])
# hsv_green = cv2.cvtColor(green, cv2.COLOR_BGR2HSV)
# print(hsv_green)
import cv2
import numpy as np

img = cv2.imdecode(np.fromfile("D:/aiImg/input/000001.jpg", dtype=np.uint8), -1)


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

print(circle_point)






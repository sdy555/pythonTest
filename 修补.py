# import numpy as np
# import cv2 as cv
# import sys
#
# from dask.rewrite import args
#
#
# class Sketcher:
#     def __init__(self, windowname, dests, colors_func):
#         self.prev_pt = None
#         self.windowname = windowname
#         self.dests = dests
#         self.colors_func = colors_func
#         self.dirty = False
#         self.show()
#         # 监听鼠标事件的触发钩子
#         cv.setMouseCallback(self.windowname, self.on_mouse)
#
#     # 显示图像和蒙版
#     def show(self):
#         cv.imshow(self.windowname, self.dests[0])
#         cv.imshow(self.windowname + ": mask", self.dests[1])
#
#     # 鼠标触发函数
#     def on_mouse(self, event, x, y, flags, param):
#         pt = (x, y)
#         # 鼠标左键按下，记录坐标
#         if event == cv.EVENT_LBUTTONDOWN:
#             self.prev_pt = pt
#         # 鼠标左键谈起，清空坐标
#         elif event == cv.EVENT_LBUTTONUP:
#             self.prev_pt = None
#         # 鼠标左键按下并拖拽绘制
#         if self.prev_pt and flags & cv.EVENT_FLAG_LBUTTON:
#             for dst, color in zip(self.dests, self.colors_func()):
#                 # 在图像和mask上绘制白色线条
#                 cv.line(dst, self.prev_pt, pt, color, 5)
#             self.dirty = True
#             self.prev_pt = pt
#             self.show()
#
#
# def main():
#
#     print("Usage: python inpaint <image_path>")
#     print("Keys: ")
#     print("t - inpaint using FMM")
#     print("n - inpaint using NS technique")
#     print("r - reset the inpainting mask")
#     print("ESC - exit")
#
#     # 读取测试图像
#     img = cv.imread('D:/aiImg/huifu.png', cv.IMREAD_COLOR)
#
#     # 判读图像是否为None
#     if img is None:
#         print('Failed to load image file: {}'.format(args["image"]))
#         return
#
#     # 使用深度拷贝创建一个愿图像副本
#     img_mask = img.copy()
#     # 创建一个全黑的mask，用来显示人为绘制的图像缺失区域
#     inpaintMask = np.zeros(img.shape[:2], np.uint8)
#     # 使用Opencv创建能够绘制缺失区域的草图
#     sketch = Sketcher('image', [img_mask, inpaintMask], lambda : ((255, 255, 255), 255))
#
#     while True:
#         ch = cv.waitKey()
#         # 根据不同的按键触发不同的效果
#         if ch == 27:
#             break
#         if ch == ord('t'):
#             # FMM算法
#             res = cv.inpaint(src=img_mask, inpaintMask=inpaintMask, inpaintRadius=3, flags=cv.INPAINT_TELEA)
#             cv.imshow('Inpaint Output using FMM', res)
#         if ch == ord('n'):
#             # NS算法
#             res = cv.inpaint(src=img_mask, inpaintMask=inpaintMask, inpaintRadius=3, flags=cv.INPAINT_NS)
#             cv.imshow('Inpaint Output using NS Technique', res)
#         if ch == ord('r'):
#             # 恢复图像
#             img_mask[:] = img
#             # 清空mask图
#             inpaintMask[:] = 0
#             sketch.show()
#
#     print('Completed')
#
#
# if __name__ == '__main__':
#     main()
#     cv.destroyAllWindows()
#
import cv2
import numpy as np
# import cv2
#
# # 读取图片
# img = cv2.imread('D:/aiImg/huifu.png')
# # 图像转换为灰度图像
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# # 灰度二值化
# _, mask = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 100, 255, cv2.THRESH_BINARY_INV)
# # _,mask = cv2.threshold(gray,10,255,cv2.THRESH_BINARY_INV)
# # mask1 = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
# # 第一种方法 img:输入图像；mask为单通道二值图掩模，非零位置为需要修复的地方；3:领域大小，在修复
# # 开路中图像素的范围
# # dst = cv2.inpaint(img,mask,10,cv2.INPAINT_TELEA)
# # 第二种方法 INPAINT_NS
# # 区域内的灰度值变化最小。
# dst = cv2.inpaint(img, mask, 10, cv2.INPAINT_NS)  # 3:领域大小
# cv2.imshow('img0', img)
# # cv2.imshow('img10',mask1)
# cv2.imshow('img1', mask)
# cv2.imshow('img2', dst)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# import cv2
# from matplotlib import pyplot as plt
#
# src_image = cv2.imread("D:/aiImg/huifu.png")
# print(src_image.shape)
# src_fig = plt.figure()
# plt.title("src_image")
# plt.imshow(src_image)
#
# gray_image = cv2.cvtColor(src_image, cv2.COLOR_RGB2GRAY)
# plt.figure()
# plt.title("gray_image")
# plt.imshow(gray_image)  # 二值化
#
# ret, binary = cv2.threshold(gray_image, 245, 255, cv2.THRESH_BINARY)  # 对灰度图进行二值化，binary是返回图像
# plt.figure()
# plt.title("binary")
# plt.imshow(binary)  # 进行开操作，去除噪音小点。
#
# kernel_3X3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # 构造卷积核
# binary_after_open = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_3X3)  # 形态学
# plt.figure()
# plt.title("binary_after_open")
# plt.imshow(binary_after_open)
#
# restored = cv2.inpaint(src_image, binary_after_open, 9, cv2.INPAINT_NS)
# restored_fig = plt.figure()
# plt.title("restored")
# plt.imshow(restored)
# plt.show()

# import cv2
# import numpy as np
# # Load the image
# image = cv2.imread('D:/aiImg/huifu.png', -1)
# (hei,wid,_) = image.shape
# #Grayscale and blur the image
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# blur = cv2.GaussianBlur(gray, (3,3), 0)
# #Threshold the image
# thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
# #Retrieve contours
# contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# #Create box-list
# box = []
# # Get position (x,y), width and height for every contour
# for c in contours:
#     x, y, w, h = cv2.boundingRect(c)
#     box.append([x,y,w,h])
#     # Create separate lists for all values
#     heights = []
#     widths = []
#     xs = []
#     ys = []
# # Store values in lists
# for b in box:
#     heights.append(b[3])
#     widths.append(b[2])
#     xs.append(b[0])
#     ys.append(b[1])
# # Retrieve minimum and maximum of lists
# min_height = np.min(heights)
# min_width = np.min(widths)
# min_x = np.min(xs)
# min_y = np.min(ys)
# max_y = np.max(ys)
# max_x = np.max(xs)
# # Retrieve height where y is maximum (edge at bottom, last row of table)
# for b in box:
#     if b[1] == max_y:
#         max_y_height = b[3]
# # Retrieve width where x is maximum (rightmost edge, last column of table)
# for b in box:
#     if b[0] == max_x:
#          max_x_width = b[2]
# # Obtain horizontal lines mask
# horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50,1))
# horizontal_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
# horizontal_mask = cv2.dilate(horizontal_mask, horizontal_kernel, iterations=9)
# # Obtain vertical lines mask
# vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,50))
# vertical_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
# vertical_mask= cv2.dilate(vertical_mask, vertical_kernel, iterations=9)
# # Bitwise-and masks together
# result = 255 - cv2.bitwise_or(vertical_mask, horizontal_mask)
# #Cropping the image to the table size
# crop_img = result[(min_y+5):(max_y+max_y_height), (min_x):(max_x+max_x_width+5)]
# #Creating a new image and filling it with white background
# img_white = np.zeros((hei, wid), np.uint8)
# img_white[:, 0:wid] = (255)
# #Retrieve the coordinates of the center of the image
# x_offset = int((wid - crop_img.shape[1])/2)
# y_offset = int((hei - crop_img.shape[0])/2)
# #Placing the cropped and repaired table into the white background
# img_white[ y_offset:y_offset+crop_img.shape[0], x_offset:x_offset+crop_img.shape[1]] = crop_img
# #Viewing the result
# cv2.imshow('Result', img_white)
# cv2.waitKey()
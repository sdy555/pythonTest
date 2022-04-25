#4.直方图均衡化
from skimage import io,data,exposure
import matplotlib.pyplot as plt
#img=data.moon()  #使用数据库里面自带的图片
filename = 'D:/aiImg/588.jpg'# 图片文件的路径
img = io.imread(filename)  # 使用imread读取图像，当使用imread时需要调用io
#plt.figure("Thin layer",figsize=(8,8))   #调整图片尺寸
#arr=img.flatten() ##默认按行的方向降维

plt.subplot(121)
plt.imshow(img,plt.cm.gray) #原始图像

# plt.subplot(222)
# plt.hist(arr, bins=100,edgecolor='None',facecolor='red') #原始图像直方图
# 当bin为整数时，则等于柱子的个数，有bin + 1个边（256+1）。
# edgecolor：是柱子边界的颜色。
# facecolor： 是柱子的颜色。

img1=exposure.equalize_hist(img) #进行直方图均衡化
arr1=img1.flatten() #返回数组折叠成一维的副本
plt.subplot(122)
plt.imshow(img1,plt.cm.gray) #均衡化图像
# plt.subplot(224)
# plt.hist(arr1, bins=100,edgecolor='None',facecolor='blue') #均衡化直方图
plt.show()

io.imshow(img1)   #io读入图片
# io.imsave('3.jpg',img1)  #保存图片，moon.jpg指的是保存路径和名称，gam1指的是需要保存的数组变量

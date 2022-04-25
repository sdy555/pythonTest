import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from skimage import data, io , img_as_float
from skimage import exposure


matplotlib.rcParams['font.size'] = 8


def plot_img_and_hist(image, axes, bins=256):
    """Plot an image along with its histogram and cumulative histogram.
    绘制图像及其直方图和累积直方图。
    """
    image = img_as_float(image)
    ax_img, ax_hist = axes
    ax_cdf = ax_hist.twinx()

    # Display image 显示图像
    ax_img.imshow(image, cmap=plt.cm.gray)
    ax_img.set_axis_off()

    # Display histogram 显示直方图状图
    ax_hist.hist(image.ravel(), bins=bins, histtype='step', color='black')
    ax_hist.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax_hist.set_xlabel('Pixel intensity')
    ax_hist.set_xlim(0, 1)
    ax_hist.set_yticks([])

    # Display cumulative distribution  显示累积分布
    # 累积分布：cumulative_distribution：（图像，nbins = 256）
    img_cdf, bins = exposure.cumulative_distribution(image, bins)
    ax_cdf.plot(bins, img_cdf, 'r')
    ax_cdf.set_yticks([])

    return ax_img, ax_hist, ax_cdf


# Load an example image
img = io.imread('D:/aiImg/img/9.jpg')

# Contrast stretching 对比度拉伸
p2, p98 = np.percentile(img, (2, 98))
img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98))

# Equalization 直方图均衡化
img_eq = exposure.equalize_hist(img)

# Adaptive Equalization  自适应均衡
# equalize_adapthist（图像，kernel_size =无，clip_limit = 0.01，nbins = 256）
img_adapteq = exposure.equalize_adapthist(img, clip_limit=0.03)

# Display results
fig = plt.figure(figsize=(8, 5))
axes = np.zeros((2, 4), dtype=np.object)
axes[0, 0] = fig.add_subplot(2, 4, 1)
for i in range(1, 4):
    axes[0, i] = fig.add_subplot(2, 4, 1+i, sharex=axes[0,0], sharey=axes[0,0])
for i in range(0, 4):
    axes[1, i] = fig.add_subplot(2, 4, 5+i)

ax_img, ax_hist, ax_cdf = plot_img_and_hist(img, axes[:, 0])# 低对比度图像
ax_img.set_title('Low contrast image')

y_min, y_max = ax_hist.get_ylim()
ax_hist.set_ylabel('Number of pixels') # 图像像素的数量
ax_hist.set_yticks(np.linspace(0, y_max, 5))

ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_rescale, axes[:, 1]) # 对比拉深度
ax_img.set_title('Contrast stretching')
# io.imsave('duibi.png',img_rescale)
ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_eq, axes[:, 2]) # 直方图均衡化
ax_img.set_title('Histogram equalization')
# io.imsave('duibi1.png',img_eq)
ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_adapteq, axes[:, 3]) # 自适应均衡
ax_img.set_title('Adaptive equalization')

ax_cdf.set_ylabel('Fraction of total intensity')
ax_cdf.set_yticks(np.linspace(0, 1, 5))

# prevent overlap of y-axis labels
fig.tight_layout()
plt.show()
import cv2 as cv


img_cat = cv.imread("D:/aiImg/cat.jpg")
img_dog = cv.imread("D:/aiImg/dog.jpg")

img_cat = cv.resize(img_dog,(500,414))
img_dog = cv.resize(img_dog,(500,414))
# print(img_dog.shape)

# res = cv.resize(img_cat, (0, 0), fx=4, fy=4)


import matplotlib.pyplot as plt
# print(plt.imshow(res))

res = cv.addWeighted(img_cat, 1.2, img_dog, 0.6, 0)


plt.imshow(res)
plt.show()
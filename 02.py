import cv2 as cv

img = cv.imread("D:/aiImg/cat.jpg",cv.IMREAD_GRAYSCALE)

def cv_show(name,img):
    cv.imshow(name,img)
    cv.waitKey(0)
    cv.destroyAllWindows()


cat = cv_show('he',img)

print(cat.size)




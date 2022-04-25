import time
import cv2
from os import listdir
from watchdog.events import *
from watchdog.observers import Observer
from skimage.exposure import match_histograms

imgType_list = {'jpg', 'bmp', 'png', 'jpeg', 'jfif'}


def piliang(image_reade, img_output):
    data_base_dir = image_reade  # 输入文件夹的路径
    outfile_dir = img_output  # 输出文件夹的路径
    reference = cv2.imread('D:/aiImg/yibiao.jpg')  # 目标图像
    processed_number = 0  # 统计处理图片的数量

    for file in os.listdir(data_base_dir):  # 遍历目标文件夹图片
        read_img_name = data_base_dir + '//' + file.strip()  # 取图片完整路径
        image = cv2.imread(read_img_name)  # 读入图片

        while (1):
            matched = match_histograms(image, reference, channel_axis=-1)
            # cv2.imshow("demo", matched)
            # k = cv2.waitKey(1)
            # if k == 13:  # 按回车键确认处理、保存图片到输出文件夹和读取下一张图片
            processed_number += 1
            out_img_name = outfile_dir + '//' + file.strip()
            cv2.imwrite(out_img_name, matched)
            print("已处理的照片数为", processed_number)
            print("按enter键以确保您的操作并处理下一张图片")
            break


def delete(url):
    my_path = url
    for file_name in listdir(my_path):
        os.remove(my_path + "\\" + file_name)


def checkImg(event):
    split = format(event.src_path).split(".")
    fileType = split[-1]
    if fileType in imgType_list:
        imgPath = format(event.src_path)
        pattern = re.compile(r'([^<>/\\\|:""\*\?]+)\.\w+$')
        fileName = pattern.findall(imgPath)
        print("file created:{0}".format(event.src_path), fileType, fileName)

        img = cv2.imread(imgPath.replace("\n", ""))
        cv2.imwrite(r'D:\aiImg\input\\' + fileName[0] + "." + fileType, img)  # 监控文件地址
        print("file created:{0}".format(event.src_path), fileType, fileName)
        piliang('D:/aiImg/input', 'D:/aiImg/output')         # 输入文件路径，输出文件路径
        delete('D:/aiImg/input')                           # 读入文件路径


class FileEventHandler(FileSystemEventHandler):

    def __init__(self):
        FileSystemEventHandler.__init__(self)

    def on_moved(self, event):
        if event.is_directory:
            print("directory moved from {0} to {1}".format(event.src_path, event.dest_path))
        else:
            print("file moved from {0} to {1}".format(event.src_path, event.dest_path))

    def on_created(self, event):
        if event.is_directory:
            print("directory created:{0}".format(event.src_path))
        else:
            checkImg(event)

    def on_deleted(self, event):
        if event.is_directory:
            print("directory deleted:{0}".format(event.src_path))
        else:
            print("file deleted:{0}".format(event.src_path))

    def on_modified(self, event):
        if event.is_directory:
            print("directory modified:{0}".format(event.src_path))
        else:
            print("file modified:{0}".format(event.src_path))


if __name__ == "__main__":
    observer = Observer()
    event_handler = FileEventHandler()
    observer.schedule(event_handler, r"D:\aiImg\in", True)  # 监控文件夹地址
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

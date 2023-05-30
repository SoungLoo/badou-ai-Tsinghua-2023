# RBG转灰度图和二值化

# 导入相关的依赖
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

filePath = "/Users/lusong/Documents/documents/study/AI/lenna.png"

# (1)手工实现rbg灰度化
# 读取原始图片，图片文件的路径要正确
img = cv2.imread(filePath)
# 获取图片的信息，high & wide   img.shape[:0] ==> 获取的是high      img.shape[:1] ==> 获取的是wide
h, w = img.shape[:2]
# 创建一张和当前图片大小一样的单通道的图片
img_gray = np.zeros([h, w], img.dtype)
# 遍历原始图片的high & wide
for i in range(h):
    for j in range(w):
        # 拿到当前high & wide中的RGB坐标
        m = img[i, j]
        # 手工转gray，转的时候需要注意openCV拿到图片的是BGR，而不是通用的RGB
        img_gray[i, j] = int(m[0]*0.11 + m[1]*0.59 + m[2]*0.3)
# 输出灰度图
print("=========image show gray==========")
print(img_gray)
# subplot主要定义图片输出展示的位置，3,3,1 表示创建3行3列的画板，放在第一个位置上
plt.subplot(3, 3, 1)
# 输出灰度图
plt.imshow(img_gray, cmap='gray')


# (2)手工实现二值化
# 通过上面实现的灰度图，再实现二值化
# 创建一张和当前灰度图大小一样的单通道图片
img_binary = np.zeros([h, w], img_gray.dtype)
# 获取灰度图的high & wide信息
grayH, grayW = img_gray.shape
# 遍历获取
for i in range(grayH):
    for j in range(grayW):
        # 二值图，因为灰度化之后，看打印出来的结果，基本都是0~255之间的灰度，所以这里取中间
        # 判断，如果小于等于127，则为0，否则为255
        if (img_gray[i, j] <= 127):
            img_binary[i, j] = 0
        # 大于0.5的，则为1
        else:
            img_binary[i, j] = 255
# 输出二值图
print("=========image show binary==========")
print(img_binary)
# subplot主要定义图片输出展示的位置
plt.subplot(3, 3, 2)
# 输出二值图
plt.imshow(img_binary, cmap='gray')


# (3)手工实现二值化，归1化处理
# 通过上面实现的灰度图，再实现二值化
# 创建一张和当前灰度图大小一样的单通道图片
img_binaryRe = np.zeros([h, w], img_gray.dtype)
# 获取灰度图的high & wide信息
grayHRe, grayWRe = img_gray.shape
# 遍历获取
for i in range(grayHRe):
    for j in range(grayWRe):
        # 拿到灰度图之后，这里做归1化处理，也就是除以255，除以255得到[0,1]之间的值，如果
        # 小于等于0.5，则记为0，否则记为255
        if (img_gray[i, j] / 255 <= 0.5):
            img_binaryRe[i, j] = 0
        # 大于0.5的，则为1
        else:
            img_binaryRe[i, j] = 255
# 输出二值图
print("=========image show resolution==========")
print(img_binaryRe)
# subplot主要定义图片输出展示的位置
plt.subplot(3, 3, 3)
# 输出二值图
plt.imshow(img_binaryRe, cmap='gray')


# (4)RGB转灰度图，调用API实现    调用rgb2gray
img_grayApi1 = rgb2gray(img)
print("=========image show rgb2gray==========")
print(img_grayApi1)
plt.subplot(3, 3, 4)
plt.imshow(img_grayApi1, cmap='gray')


# (5)RGB转灰度图，调用API实现    调用OpenCV的API
img_grayApi2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print("=========image show COLOR_BGR2GRAY==========")
print(img_grayApi2)
plt.subplot(3, 3, 5)
plt.imshow(img_grayApi2, cmap='gray')


# (6)二值化，调用API实现
img_binaryApi = np.where(img_gray >= 127, 1, 0)
print("=========image show where==========")
print(img_binaryApi)
plt.subplot(3, 3, 6)
plt.imshow(img_binaryApi, cmap='gray')

# (7)输出原图 使用plt读取
# 注意，这里如果用cv2去读取原图，然后直接输出，会有问题，因为OpenCV默认是BGR，需要转换成RGB
img_origin = plt.imread(filePath)
print("=========image show origin==========")
print(img_origin)
plt.subplot(3, 3, 7)
plt.imshow(img_origin)

# (8)输出原图  使用CV2读取，并且进行BGR转RGB
img_originCv = cv2.imread(filePath)
# 注意，这里如果用cv2去读取原图，然后直接输出，会有问题，因为OpenCV默认是BGR，需要转换成RGB
img_originChange = cv2.cvtColor(img_originCv, cv2.COLOR_BGR2RGB)
print("=========image show img_originChange==========")
print(img_originChange)
plt.subplot(3, 3, 8)
plt.imshow(img_originChange)

# (9)输出原图  使用CV2读取，但未进行BGR转RGB
img_originCv = cv2.imread(filePath)
# 注意，这里如果用cv2去读取原图，然后直接输出，会有问题，因为OpenCV默认是BGR，需要转换成RGB
# img_origin1 = cv2.cvtColor(img_originChange, cv2.COLOR_BGR2RGB)
print("=========image show img_originUnChange==========")
print(img_originCv)
plt.subplot(3, 3, 9)
plt.imshow(img_originCv)


# 输出显示图片
plt.show()


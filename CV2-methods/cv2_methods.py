# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 09:05:08 2021

@author: 1618047
"""

# 
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

path_vid = r'D:/abyss_of_work/main/Aleksandr_G.K/summer_practice/projects/heart_filtres/heart.MOV' # путь к видео
path_img = r'D:/abyss_of_work/main/Aleksandr_G.K/2021/Image-processing/CV2-methods/frames/frame0.jpg' # путь к фрейму

'''
# ==================================#=========================================
# Открыть картинку в отдельном окне #
# ==================================#
'''

# image = cv2.imread(path_img) # Задаёт путь к фрейму в переменную

# cv2.imshow("image", image) # открываем окно с изображением
# cv2.waitKey(0) # ключ удержания на бесконечность (поставим вместо 0 цифру, зададим удержание на время в мс)
# cv2.destroyAllWindows()
# ============================================================================

'''
# ==============================#=============================================
# Открыть видо в отдельном окне #
# ==============================#
'''

# video = cv2.VideoCapture(path_vid) # Задаёт путь к нашему видео в переменную

# while(True):
#     ref, frame = video.read() # ref - булевское значение 1 или 0. frame - наше видео. Функция video.read() читает видео
    
#     cv2.imshow('frame', frame) # выдаёт окно с видео
    
#     k = cv2.waitKey(30) & 0xff # команда, чтобы при нажатии esc видео закрылось
#     if k == 27: # хз что это, но это нужно
#         break

# video.release() # закрываем снова какими-то непонятными командами, но они тоже нужны
# cv2.destroyAllWindows() # и это тоже нужно
# ============================================================================

'''
# ======================#=====================================================
# Открыть видео с вебки #
# ======================#
'''

# cap = cv2.VideoCapture(0)
# while(1):
#     _, img = cap.read()
#     cv2.imshow('img', img)
    
#     k = cv2.waitKey(30) & 0xff
#     if k == 27:
#         break
# cap.release()
# cv2.destroyAllWindows()

# ============================================================================

'''
# =========================#==================================================
# Открыть видео с телефона #
# =========================#
'''

# import imutils
# import requests
# # Replace the below URL with your own. Make sure to add "/shot.jpg" at last.
# url = "http://190.000.0.000:8000/shot.jpg"
# cap = cv2.VideoCapture(url)

# while True:
#     img_resp = requests.get(url)
#     img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
#     img = cv2.imdecode(img_arr, -1)
#     img = imutils.resize(img, width=1000, height=1800)
    
#     # Сдесь могли быть ваши фильтры
    
#     cv2.imshow("Android_cam", img)
  
#     # Press Esc key to exit
#     if cv2.waitKey(1) == 27:
#         break
  
# cv2.destroyAllWindows()
# ============================================================================

'''
# =========================#==================================================
# Разбиение видео на кадры #
# =========================#
'''

# video = cv2.VideoCapture(path_vid)
# count = 0
# directory = r'D:\abyss_of_work\main\Aleksandr_G.K\2021\Image-processing\CV2-methods\frames' # задаём куда сохранять фреймы
# os.chdir(directory) # переходим в заданную директорию

# while video.isOpened(): # цикл работает пока эткрыто видео
#     _, vid = video.read()
#     print(count)
#     cv2.imshow('video', vid)
#     cv2.imwrite("frame%d.jpg" % count, vid) # сохраняем каждый кадр 
#     count = count + 1
    
#     if cv2.waitKey(10) & 0xFF == ord('q'):
#          break
# video.release()
# cv2.destroyAllWindows()
# ============================================================================

'''
# =====================#======================================================
# Перевод в чёрнобелое #
# =====================#
'''

# image = cv2.imread('toad.jpg')

# gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# cv2.imshow('Grayscale', gray_image)
# cv2.waitKey(0) 
# cv2.destroyAllWindows()

# # Можно по другому

# image = cv2.imread('toad.jpg', 0) #<<< просто ставим 0
# cv2.imshow('Grayscale Image', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# ============================================================================

'''
# =======================================#====================================
# Наложение одного изображения на другое #
# =======================================#
'''

# mask = cv2.imread('mask512.jpg')
# img = cv2.imread(path_img)

# weightedSum = cv2.addWeighted(img, 0.5, mask, 0.4, 0) # 2-й параметр - яркость 1 изображения, 4-й - яркость изображения внутри маски
# cv2.imshow('Weighted Image', weightedSum)

# if cv2.waitKey(0) & 0xff == 27: 
#     cv2.destroyAllWindows()
# ============================================================================

'''
# ================#===========================================================
# Масштабирование #
# ================#
'''

# image = cv2.imread('toad.jpg')

# height = 720
# weight = 1280

# half = cv2.resize(image, (weight, height))
# cv2.imshow("half", half)
# cv2.waitKey(0) 
# cv2.destroyAllWindows()

# ============================================================================

'''
# ========================#===================================================
# Метод erode или эррозия #
# ========================#
'''

# image = cv2.imread('toad.jpg')

# height = 720
# weight = 1280
# half = cv2.resize(image, (weight, height))

# y_param = 5
# x_param = 5

# kernel = np.ones((y_param, x_param), np.uint8)
# image = cv2.erode(half, kernel)
# cv2.imshow("image", image)
# cv2.waitKey(0) 
# cv2.destroyAllWindows()

# ============================================================================
'''
# ==========================================#=================================
# Размытие. Удаление шума. Удаление деталей # 
# ==========================================#

Используется для обработки перед наложением основных фильтров
'''

# image = cv2.imread('toad.jpg')

# cv2.imshow('image', image)

# # Gaussian Blur # общее размытие
# Gaussian = cv2.GaussianBlur(image, (7, 7), 0)
# cv2.imshow('Gaussian Blurring', Gaussian)

# # Median Blur 
# median = cv2.medianBlur(image, 5)
# cv2.imshow('Median Blurring', median)

# # Bilateral Blur # Размытие с сохранением границ
# bilateral = cv2.bilateralFilter(image, 9, 75, 75)
# cv2.imshow('Bilateral Blurring', bilateral)

# cv2.waitKey(0)
# cv2.destroyAllWindows()
# ============================================================================

'''
# ====================#=======================================================
# Поворот изображения #
# ====================#
'''

# img = cv2.imread('toad.jpg')
# (rows, cols) = img.shape[:2]

# angle = 45
# zoom = 1

# M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, zoom) 
# res = cv2.warpAffine(img, M, (cols, rows))
# cv2.imshow('res', res)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

# ============================================================================

'''
# ===============================#============================================
# Фильтр Canny, выделение краёв #
# ===============================#
'''

# img = cv2.imread('toad.jpg')

# img_edges = cv2.Canny(img, 100, 200)
# cv2.imshow('img_edges', img_edges)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

# ============================================================================

'''
# =====================#======================================================
# Erosion and Dilation #
# =====================#

Эрозия:
Это полезно для удаления небольших белых шумов.
Используется для разъединения двух связанных объектов и т.д.

Расширение:
В таких случаях, как удаление шума, за эрозией следует расширение. Потому что эрозия убирает белые шумы,
но она также сжимает наш объект. Итак, мы расширяем его. Поскольку шум ушел, они не вернутся,
но площадь нашего объекта увеличивается.
Это также полезно для соединения сломанных частей объекта. 
'''

# img = cv2.imread('toad.jpg')

# kernel = np.ones((5,5), np.uint8)
# img_erosion = cv2.erode(img, kernel, iterations=1)
# img_dilation = cv2.dilate(img, kernel, iterations=1)
# er_dil = cv2.dilate(img_erosion, kernel, iterations=1)

# cv2.imshow('Input', img)
# cv2.imshow('Erosion', img_erosion)
# cv2.imshow('Dilation', img_dilation)
# cv2.imshow('er_dil', er_dil)
 
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# ============================================================================

'''
# =========================================#==================================
# Анализ изображений  с помощью гистограмм #
# =========================================#

images: it is the source image
channels: it is the index of channel for which we calculate histogram.
For grayscale image, its value is [0]
and color image, you can pass [0],[1] or [2] to calculate histogram of blue, green or red channel respectively.
mask: mask image. To find histogram of full image, it is given as “None”.
histSize: this represents our BIN count. For full scale, we pass [256].
ranges: this is our RANGE. Normally, it is [0,256].
'''
# imgG = cv2.imread('toad.jpg',0)
# img = cv2.imread('toad.jpg')

# # Gray histogram
# histg = cv2.calcHist([imgG],[0],None,[256],[0,256])
# plt.figure(figsize=(7, 5), dpi=300)
# plt.plot(histg)
# plt.show()

# # BGR histogram
# plt.figure(figsize=(7, 5), dpi=300)
# color = ('b','g','r')
# for i,col in enumerate(color):
#     histr = cv2.calcHist([img],[i],None,[256],[0,256])
#     plt.plot(histr,color = col)
#     plt.xlim([0,256])
# plt.show()
 
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# ============================================================================

'''
# ==========#=================================================================
# Threshold #
# ==========#
В компьютерном зрении этот метод определения порога применяется к изображениям в градациях серого.
Поэтому изначально изображение должно быть преобразовано в цветовое пространство оттенков серого. 

cv2.THRESH_BINARY: если интенсивность пикселей больше установленного порога, устанавливается значение 255, в противном случае устанавливается значение 0 (черный).
cv2.THRESH_BINARY_INV: инвертированный или противоположный регистр cv2.THRESH_BINARY.
cv.THRESH_TRUNC: если значение интенсивности пикселя больше порогового значения, оно обрезается до порогового значения. Значения пикселей устанавливаются такими же, как пороговое значение. Все остальные значения остаются прежними.
cv.THRESH_TOZERO: Интенсивность пикселей установлена на 0, для всех пикселей яркость меньше порогового значения.
cv.THRESH_TOZERO_INV: инвертированный или противоположный регистр cv2.THRESH_TOZERO.
'''

# img = cv2.imread('toad.jpg', 0)

# ret, thresh1 = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)
# ret, thresh2 = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY_INV)
# ret, thresh3 = cv2.threshold(img, 120, 255, cv2.THRESH_TRUNC)
# ret, thresh4 = cv2.threshold(img, 120, 255, cv2.THRESH_TOZERO)
# ret, thresh5 = cv2.threshold(img, 120, 255, cv2.THRESH_TOZERO_INV)

# cv2.imshow('Binary Threshold', thresh1)
# cv2.imshow('Binary Threshold Inverted', thresh2)
# cv2.imshow('Truncated Threshold', thresh3)
# cv2.imshow('Set to 0', thresh4)
# cv2.imshow('Set to 0 Inverted', thresh5)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

# ============================================================================

'''
# ===================#========================================================
# Adaptive Threshold #
# ===================#

Адаптивная пороговая обработка - это метод,
при котором пороговое значение рассчитывается для небольших регионов.
Это приводит к разным пороговым значениям для разных регионов в отношении изменения освещения.
'''
# img = cv2.imread('toad.jpg', 0)

# thresh1 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
#                                           cv2.THRESH_BINARY, 199, 5)
  
# thresh2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                           cv2.THRESH_BINARY, 199, 5)
# cv2.imshow('Adaptive Mean', thresh1)
# cv2.imshow('Adaptive Gaussian', thresh2)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

# ============================================================================

'''
# ==================#=========================================================
# Otsu Thresholding #
# ==================#

В Otsu Thresholding значение порога не выбирается, а определяется автоматически.
Рассматривается бимодальное изображение (два различных значения изображения).
Созданная гистограмма содержит два пика. Итак, общим условием будет выбор порогового значения,
которое находится в середине обоих пиковых значений гистограммы.

Syntax: cv2.threshold(source, thresholdValue, maxVal, thresholdingTechnique)
thresholdValue: Значение порога, ниже и выше которого значения пикселей будут соответственно изменяться.
maxVal: максимальное значение, которое может быть присвоено пикселю.
thresholdingTechnique: Тип применяемого порогового значения.
'''

# img = cv2.imread('toad.jpg', 0)

# ret, thresh1 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + 
#                                             cv2.THRESH_OTSU)
# cv2.imshow('Otsu Threshold', thresh1)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

# ============================================================================

'''
# ============================================================#===============
# Convert an image from one color space to another. Gray, HSV #
# ============================================================#

Syntax: cv2.cvtColor(src, code[, dst[, dstCn]])

src: это изображение, цветовое пространство которого необходимо изменить.
code: это код преобразования цветового пространства.
dst: это выходное изображение того же размера и глубины, что и изображение src.
Это необязательный параметр.
dstCn: это количество каналов в конечном изображении.
Если параметр равен 0, то количество каналов определяется автоматически из src и кода.
Это необязательный параметр. 
'''

# img = cv2.imread('toad.jpg')

# image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow('GRAY', image)

'''
HSV color space
'''

# image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# cv2.imshow('HSV', image)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

# ============================================================================

'''
# =======================================================#====================
# Цветовой фильтр. Filter Color, вырезание нужного цвета #
# =======================================================#
'''

# cap = cv2.VideoCapture(0)
 
# while(1):
#     _, frame = cap.read()
    
#     frame = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 15)
#     # It converts the BGR color space of image to HSV color space
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
#     # Threshold of blue in HSV space
#     lower_blue = np.array([60, 35, 140])
#     upper_blue = np.array([180, 255, 255])
    
#     # preparing the mask to overlay
#     mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
#     # The black region in the mask has the value of 0,
#     # so when multiplied with original image removes all non-blue regions
#     result = cv2.bitwise_and(frame, frame, mask = mask)
    
#     cv2.imshow('frame', frame)
#     cv2.imshow('mask', mask)
#     cv2.imshow('result', result)
     
#     k = cv2.waitKey(30) & 0xff
#     if k == 27:
#         break
# cv2.destroyAllWindows()
# cap.release()
    
# ============================================================================

'''
# ==========================#=================================================
# Denoising. Как убарть шум #
# ==========================#
Syntax: cv2.fastNlMeansDenoisingColored( P1, P2, float P3, float P4, int P5, int P6)
P1 – Source Image Array
P2 – Destination Image Array
P3 – Size in pixels of the template patch that is used to compute weights.
P4 – Size in pixels of the window that is used to compute a weighted average for the given pixel.
P5 – Parameter regulating filter strength for luminance component.
P6 – Same as above but for color components // Not used in a grayscale image.
'''
# img = cv2.imread('toad.jpg')
# dst = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 6, 15)

# cv2.imshow('dst', dst)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

# ============================================================================

'''
# =====================#======================================================
# Поиск контуров фигур #
# =====================#
'''

# img2 = cv2.imread('figures.jpg', cv2.IMREAD_COLOR)
# img = cv2.imread('figures.jpg', 0)
# font = cv2.FONT_HERSHEY_COMPLEX

# _, threshold = cv2.threshold(img, 110, 255, cv2.THRESH_BINARY)

# # Detecting contours in image.
# contours, _= cv2.findContours(threshold, cv2.RETR_TREE,
#                                cv2.CHAIN_APPROX_SIMPLE)

# # Going through every contours found in the image.
# for cnt in contours :
  
#     approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True)
  
#     # draws boundary of contours.
#     cv2.drawContours(img2, [approx], 0, (0, 0, 255), 5) 
  
#     # Used to flatted the array containing
#     # the co-ordinates of the vertices.
#     n = approx.ravel() 
#     i = 0
  
#     for j in n :
#         if(i % 2 == 0):
#             x = n[i]
#             y = n[i + 1]
  
#             # String containing the co-ordinates.
#             string = str(x) + " " + str(y) 
  
#             if(i == 0):
#                 # text on topmost co-ordinate.
#                 cv2.putText(img2, "Arrow tip", (x, y),
#                                 font, 0.5, (255, 0, 0)) 
#             else:
#                 # text on remaining co-ordinates.
#                 cv2.putText(img2, string, (x, y), 
#                           font, 0.5, (0, 255, 0)) 
#         i = i + 1
  
# # Showing the final image.
# cv2.imshow('image2', img2) 
  
# # Exiting the window if 'q' is pressed on the keyboard.
# if cv2.waitKey(0) & 0xFF == ord('q'): 
#     cv2.destroyAllWindows()

# ============================================================================

'''
# ====================#=======================================================
# Log Transformations #
# ====================#
'''
# img = cv2.imread('toad.jpg',0)

# # Apply log transform.
# c = 255/(np.log(1 + np.max(img)))
# log_transformed = c * np.log(1 + img)
  
# # Specify the data type.
# log_transformed = np.array(log_transformed, dtype = np.uint8)
# cv2.imshow('log_transformed', log_transformed)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

'''
# =====================#
# Gamma Transformation #
# =====================#
'''
# img = cv2.imread('toad.jpg',0)
# # Trying 4 gamma values.
# for gamma in [0.1, 0.5, 1.2, 2.2]:
      
#     # Apply gamma correction.
#     gamma_corrected = np.array(255*(img / 255) ** gamma, dtype = 'uint8')
  
#     # Save edited images.
#     cv2.imshow('gamma_transformed'+str(gamma), gamma_corrected)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# ============================================================================

'''
# ===========================================#================================
# Background subtraction. Убираем задний фон #
# ===========================================#
'''
# cap = cv2.VideoCapture(0)
# fgbg = cv2.createBackgroundSubtractorMOG2()

# while(1):
#     ret, frame = cap.read()
 
#     fgmask = fgbg.apply(frame)
  
#     cv2.imshow('fgmask', fgmask)
#     cv2.imshow('frame',frame )
 
     
#     k = cv2.waitKey(30) & 0xff
#     if k == 27:
#         break
     
 
# cap.release()
# cv2.destroyAllWindows()
# ============================================================================

'''
# =========================#==================================================
# Morphological Operations #
# =========================#
'''
# screenRead = cv2.VideoCapture(0)
 
# while(1):
#     _, image = screenRead.read()
     
#     hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
     
#     # defining the range of masking
#     blue1 = np.array([110, 70, 60])
#     blue2 = np.array([130, 255, 255])
     
#     # initializing the mask to be
#     # convoluted over input image
#     mask = cv2.inRange(hsv, blue1, blue2)
 
#     # passing the bitwise_and over
#     # each pixel convoluted
#     res = cv2.bitwise_and(image, image, mask = mask)
     
#     # defining the kernel i.e. Structuring element
#     kernel = np.ones((5, 5), np.uint8)
     
#     # defining the opening function
#     # over the image and structuring element
#     opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
#     cv2.imshow('Mask', mask)
#     cv2.imshow('Opening', opening)
     
#     k = cv2.waitKey(30) & 0xff
#     if k == 27:
#         break
# cv2.destroyAllWindows()
# screenRead.release()
# ============================================================================

'''
# =========#==================================================================
# Gradient #
# =========#
'''
# screenRead = cv2.VideoCapture(0)

# while(1):
#     _, image = screenRead.read()

#     hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
      
#     # defining the range of masking
#     blue1 = np.array([110, 50, 50])
#     blue2 = np.array([130, 255, 255])
      
#     # initializing the mask to be
#     # convoluted over input image
#     mask = cv2.inRange(hsv, blue1, blue2)
  
#     # passing the bitwise_and over
#     # each pixel convoluted
#     res = cv2.bitwise_and(image, image, mask = mask)
      
#     # defining the kernel i.e. Structuring element
#     kernel = np.ones((5, 5), np.uint8)
      
#     # defining the gradient function 
#     # over the image and structuring element
#     gradient = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)
     
#     cv2.imshow('Gradient', gradient)

#     k = cv2.waitKey(30) & 0xff
#     if k == 27:
#         break
# cv2.destroyAllWindows()
# screenRead.release()
# ============================================================================

'''
# ===================#========================================================
# Image segmentation # (На самом деле такое себе сенментирование)
# ===================#
'''
# img = cv2.imread('toad.jpg')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  
# ret, thresh = cv2.threshold(gray, 0, 255,
#                             cv2.THRESH_BINARY_INV +
#                             cv2.THRESH_OTSU)

# # Noise removal using Morphological
# # closing operation
# kernel = np.ones((3, 3), np.uint8)
# closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE,
#                             kernel, iterations = 2)
  
# # Background area using Dialation
# bg = cv2.dilate(closing, kernel, iterations = 1)
  
# # Finding foreground area
# dist_transform = cv2.distanceTransform(closing, cv2.DIST_L2, 0)
# ret, fg = cv2.threshold(dist_transform, 0.02
#                         * dist_transform.max(), 255, 0)
# cv2.imshow('fg', fg)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# ============================================================================

'''
# ==================================#=========================================
# Line detection. Обнаружение линий #
# ==================================#
'''
img = cv2.imread('leaf.jpg')

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  
# Apply edge detection method on the image
edges = cv2.Canny(gray,50,150,apertureSize = 3)
  
# This returns an array of r and theta values
lines = cv2.HoughLines(edges,1,np.pi/180, 200)
  
# The below for loop runs till r and theta values 
# are in the range of the 2d array
for r,theta in lines[0]:
      
    # Stores the value of cos(theta) in a
    a = np.cos(theta)
  
    # Stores the value of sin(theta) in b
    b = np.sin(theta)
      
    # x0 stores the value rcos(theta)
    x0 = a*r
      
    # y0 stores the value rsin(theta)
    y0 = b*r
      
    # x1 stores the rounded off value of (rcos(theta)-1000sin(theta))
    x1 = int(x0 + 1000*(-b))
      
    # y1 stores the rounded off value of (rsin(theta)+1000cos(theta))
    y1 = int(y0 + 1000*(a))
  
    # x2 stores the rounded off value of (rcos(theta)+1000sin(theta))
    x2 = int(x0 - 1000*(-b))
      
    # y2 stores the rounded off value of (rsin(theta)-1000cos(theta))
    y2 = int(y0 - 1000*(a))
      
    # cv2.line draws a line in img from the point(x1,y1) to (x2,y2).
    # (0,0,255) denotes the colour of the line to be 
    #drawn. In this case, it is red. 
    cv2.line(img,(x1,y1), (x2,y2), (0,0,255),2)
      
# All the changes made in the input image are finally
# written on a new image houghlines.jpg
cv2.imshow('linesDetected.jpg', img)

cv2.waitKey(0)
cv2.destroyAllWindows()
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

# ==================================#=========================================
# Открыть картинку в отдельном окне #
# ==================================#

# image = cv2.imread(path_img) # Задаёт путь к фрейму в переменную

# cv2.imshow("image", image) # открываем окно с изображением
# cv2.waitKey(0) # ключ удержания на бесконечность (поставим вместо 0 цифру, зададим удержание на время в мс)
# cv2.destroyAllWindows()
# ============================================================================

# ==============================#=============================================
# Открыть видо в отдельном окне #
# ==============================#

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

# ======================#=====================================================
# Открыть видео с вебки #
# ======================#

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

# =========================#==================================================
# Разбиение видео на кадры #
# =========================#

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

# =====================#======================================================
# Перевод в чёрнобелое #
# =====================#

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

# =======================================#====================================
# Наложение одного изображения на другое #
# =======================================#

# mask = cv2.imread('mask512.jpg')
# img = cv2.imread(path_img)

# weightedSum = cv2.addWeighted(img, 0.5, mask, 0.4, 0) # 2-й параметр - яркость 1 изображения, 4-й - яркость изображения внутри маски
# cv2.imshow('Weighted Image', weightedSum)

# if cv2.waitKey(0) & 0xff == 27: 
#     cv2.destroyAllWindows()
# ============================================================================

# ================#===========================================================
# Масштабирование #
# ================#

# image = cv2.imread('toad.jpg')

# height = 720
# weight = 1280

# half = cv2.resize(image, (weight, height))
# cv2.imshow("half", half)
# cv2.waitKey(0) 
# cv2.destroyAllWindows()

# ============================================================================

# ========================#===================================================
# Метод erode или эррозия #
# ========================#

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

# ==========================================#=================================
# Размытие. Удаление шума. Удаление деталей # 
# ==========================================#
'''
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

# ====================#=======================================================
# Поворот изображения #
# ====================#

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

# ===============================#============================================
# Фильтр Canny, выделение краёв #
# ===============================#

# img = cv2.imread('toad.jpg')

# img_edges = cv2.Canny(img, 100, 200)
# cv2.imshow('img_edges', img_edges)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

# ============================================================================

# =====================#======================================================
# Erosion and Dilation #
# =====================#

'''
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


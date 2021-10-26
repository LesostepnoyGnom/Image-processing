# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 09:05:08 2021

@author: 1618047
"""

# 
import cv2
import os

path_vid = r'D:/abyss_of_work/main/Aleksandr_G.K/summer_practice/projects/heart_filtres/heart.MOV' # путь к видео
path_img = r'D:/abyss_of_work/main/Aleksandr_G.K/2021/Image-processing/CV2-methods/frames/frame0.jpg' # путь к фрейму

# ======================================#=====================================
# Как открыть картинку в отдельном окне #
# ======================================#

# image = cv2.imread(path_img) # Задаёт путь к фрейму в переменную

# cv2.imshow("image", image) # открываем окно с изображением
# cv2.waitKey(0) # ключ удержания на бесконечность (поставим вместо 0 цифру, зададим удержание на время в мс)
# cv2.destroyAllWindows()
# ============================================================================

# ==================================#=========================================
# Как открыть видо в отдельном окне #
# ==================================#

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

# ===========================#================================================
# Как разбить видео на кадры #
# ===========================#================================================

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

# =======================================#====================================
# Как наложить одно изображение на другое#
# =======================================#====================================

# mask = cv2.imread('mask512.jpg')
# img = cv2.imread(path_img)

# weightedSum = cv2.addWeighted(img, 0.5, mask, 0.4, 0) # 2-й параметр - яркость 1 изображения, 4-й - яркость изображения внутри маски
# cv2.imshow('Weighted Image', weightedSum)

# if cv2.waitKey(0) & 0xff == 27: 
#     cv2.destroyAllWindows()
# ============================================================================


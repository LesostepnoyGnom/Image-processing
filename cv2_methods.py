# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 09:05:08 2021

@author: 1618047
"""

# 
import cv2

path = r'D:/abyss_of_work/main/Aleksandr_G.K/summer_practice/projects/heart_filtres/heart.MOV'

video = cv2.VideoCapture(path) # Задаёт путь к нашему видео

# ==================================#=========================================
# Как открыть видо в отдельном окне #
# ==================================#
while(True):
    ref, frame = video.read() # ref - булевское значение 1 или 0. frame - наше видео. Функция video.read() читает видео
    
    cv2.imshow('frame', frame) # выдаёт окно с видео
    
    k = cv2.waitKey(30) & 0xff # команда, чтобы при нажатии esc видео закрылось
    if k == 27: # хз что это, но это нужно
        break

video.release() # закрываем снова какими-то непонятными командами, но они тоже нужны
cv2.destroyAllWindows() # и это тоже нужно
# ============================================================================


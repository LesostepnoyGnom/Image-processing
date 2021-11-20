# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 10:24:24 2021

@author: 1618047
"""

import cv2
from openpiv import tools, pyprocess, validation, filters, scaling

import numpy as np
import matplotlib.pyplot as plt

import imageio


path_a = r'D:\abyss_of_work\main\Aleksandr_G.K\summer_practice\projects\heart_filtres\frames\frame73.jpg'
path_b = r'D:\abyss_of_work\main\Aleksandr_G.K\summer_practice\projects\heart_filtres\frames\frame74.jpg'

img_a = cv2.imread(path_a, 0)
img_b = cv2.imread(path_b, 0)

frame_a = img_a[0:720, 100:1000]
frame_b = img_b[0:720, 100:1000]

fig,ax = plt.subplots(1,2,figsize=(12,10),dpi=300)
ax[0].imshow(frame_a,cmap=plt.cm.gray);
ax[1].imshow(frame_b,cmap=plt.cm.gray);

'''
This function allows the search area (search_area_size) in the second frame to be larger than the interrogation window in the first frame (window_size).
Also, the search areas can overlap (overlap).

The extended_search_area_piv function will return three arrays.
1. The u component of the velocity vectors
2. The v component of the velocity vectors
3. The signal-to-noise ratio (S2N) of the cross-correlation map of each vector.
The higher the S2N of a vector, the higher the probability that its magnitude and direction are correct.
'''

winsize = 64 # pixels, interrogation window size in frame A
searchsize = 70  # pixels, search area size in frame B
overlap = 30 # pixels, 50% overlap
dt = 0.02 # sec, time interval between the two frames

u0, v0, sig2noise = pyprocess.extended_search_area_piv(
    frame_a.astype(np.int32),
    frame_b.astype(np.int32),
    window_size=winsize,
    overlap=overlap,
    dt=dt,
    search_area_size=searchsize,
    sig2noise_method='peak2peak',
)

'''The function get_coordinates finds the center of each interrogation window. This will be useful later on when plotting the vector field.'''
x, y = pyprocess.get_coordinates(
    image_size=frame_a.shape,
    search_area_size=searchsize,
    overlap=overlap,
)

'''To start, lets use the function sig2noise_val to get a mask indicating which vectors have a minimum amount of S2N. Vectors below a certain threshold are substituted by NaN.'''

u1, v1, mask = validation.sig2noise_val(
    u0, v0,
    sig2noise,
    threshold = 1.05,
)

'''Another useful function is replace_outliers, which will find outlier vectors,
and substitute them by an average of neighboring vectors.
The larger the kernel_size the larger is the considered neighborhood.
This function uses an iterative image inpainting algorithm.
The amount of iterations can be chosen via max_iter.'''

u2, v2 = filters.replace_outliers(
    u1, v1,
    method='localmean',
    max_iter=3,
    kernel_size=3,
)

'''Next, we are going to convert pixels to millimeters,
and flip the coordinate system such that the origin becomes the bottom left corner of the image.'''

# convert x,y to mm
# convert u,v to mm/sec

x, y, u3, v3 = scaling.uniform(
    x, y, u2, v2,
    scaling_factor = 96.52,  # 96.52 pixels/millimeter
)

# 0,0 shall be bottom left, positive rotation rate is counterclockwise
x, y, u3, v3 = tools.transform_coordinates(x, y, u3, v3)

'''The function save is used to save the vector field to a ASCII tabular file.
The coordinates and S2N mask are also saved'''

tools.save(x, y, u3, v3, mask, 'exp1_001.txt' )

fig, ax = plt.subplots(figsize=(8,8), dpi=300)
tools.display_vector_field(
    'exp1_001.txt',
    ax=ax, scaling_factor=96.52,
    scale=60, # scale defines here the arrow length
    width=0.0035, # width is the thickness of the arrow
    on_img=True, # overlay on the image
    image_name = path_a);
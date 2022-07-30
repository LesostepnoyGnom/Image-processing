# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 12:02:05 2022

@author: 1618047
"""

import matplotlib.pyplot as plt
import os
from openpiv import tools, pyprocess, validation, filters, scaling

os.chdir("D:/abyss_of_work/main/Aleksandr_G.K/Projects/Image-processing/vortex_analysis/test_heatmaps/Re1")
num_of_images = len(os.listdir('.'))
print(num_of_images)

for i in range(num_of_images):
    
    vf_path = r'D:/abyss_of_work/main/Aleksandr_G.K/Projects/Image-processing/vortex_analysis/vector_field/vector'+str(i)+'.jpg'
    hm_path = r'D:/abyss_of_work/main/Aleksandr_G.K/Projects/Image-processing/vortex_analysis/test_heatmaps/Re1/heatmap_Re1_'+str(i)+'.jpg'
    
    vf_img = tools.imread(vf_path)
    hm_img = tools.imread(hm_path)
    
    fig,ax = plt.subplots(1,2,figsize=(12,10), dpi=300)
    ax[0].imshow(vf_img)
    ax[1].imshow(hm_img)
    
    # fig = plt.figure(figsize = (7,5), dpi=300)
    # with cbook.get_sample_data(vf_path + 'vector' + str(i) + '.jpg') as image_file:
    #     vf_img = plt.imread(image_file)
    
    # with cbook.get_sample_data(hm_path + 'heatmap_Re1_' + str(i) + '.jpg') as image_file:
    #     hm_img = plt.imread(image_file)
    
    # plt.subplot(1,2,1)
    # plt.imshow(vf_img)
    # plt.title('vector field '+str(i))
    
    # plt.subplot(1,2,2)
    # plt.imshow(hm_img)
    # plt.title('vf+hm_'+str(i))
    
    a = 'vector'+str(i)+'.jpg'
    plt.savefig(r'D:/abyss_of_work/main/Aleksandr_G.K/Projects/Image-processing/vortex_analysis/vf+hm/'+str(a), dpi = 300)
    plt.show()
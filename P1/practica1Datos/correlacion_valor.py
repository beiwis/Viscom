#####################################################################################
#
# Visión por Computador 2024 - Práctica 1
#
#####################################################################################
#
# Authors: Alejandro Perez, Jesus Bermudez, JMM Montiel
#
# Version: 1.0
#
#####################################################################################

import sys
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def main():

    img = cv.imread('salvadorTeruel.tif')
    img = cv.cvtColor(img,cv.COLOR_BGR2RGB) # Load image 1 and conversion from BGR 2 RGB
    img_gray = cv.cvtColor(img,cv.COLOR_RGB2GRAY)

    kernel_size = 21

    # Create a figure and a 2x2 grid of subplots
    fig, axs = plt.subplots(2, 2, figsize=(10, 6))
    fig.delaxes(axs[0,0])
    fig.delaxes(axs[0,1])

    # Create a subplot that spans both rows on the left
    ax_img = fig.add_subplot(2, 1, 1)
    ax_kernel = fig.add_subplot(axs[1,0])
    ax_query = fig.add_subplot(axs[1,1])
    ax_img.axis('off')
    ax_kernel.axis('off')
    ax_query.axis('off')
    
    ax_kernel.set_title('Kernel image')

    ax_img.imshow(img_gray, cmap="gray")
    coord_clicked_point = plt.ginput(1, show_clicks=False)
    k_clicked = np.array([coord_clicked_point[0][0], coord_clicked_point[0][1]])
    k_clicked = np.round(k_clicked).astype(np.uint16)
    # We draw the point with its label
    ax_img.plot(k_clicked[0], k_clicked[1], '+r', markersize=15)
    delta = int((kernel_size-1)/2)
    ax_img.plot([k_clicked[0]-delta,k_clicked[0]+delta],[k_clicked[1]-delta,k_clicked[1]-delta],'r')
    ax_img.plot([k_clicked[0]-delta,k_clicked[0]-delta],[k_clicked[1]-delta,k_clicked[1]+delta],'r')
    ax_img.plot([k_clicked[0]-delta,k_clicked[0]+delta],[k_clicked[1]+delta,k_clicked[1]+delta],'r')
    ax_img.plot([k_clicked[0]+delta,k_clicked[0]+delta],[k_clicked[1]+delta,k_clicked[1]-delta],'r')
    plt.draw()  # We update the former image without create a new context

    kernel = img_gray[k_clicked[1]-delta:k_clicked[1]+delta, k_clicked[0]-delta:k_clicked[0]+delta]
    kernel_big = cv.resize(kernel,(kernel_size*10,kernel_size*10), interpolation=cv.INTER_NEAREST)

    ax_kernel.imshow(kernel_big,cmap="gray")
    plt.draw()  # We update the former image without create a new context
    plt.waitforbuttonpress()

    while(1):
        coord_clicked_point = plt.ginput(1, show_clicks=False)
        p_clicked = np.array([coord_clicked_point[0][0], coord_clicked_point[0][1]])
        p_clicked = np.round(p_clicked).astype(np.uint16)
        # We draw the point with its label
        ax_img.clear()
        ax_img.axis('off')
        ax_img.imshow(img_gray,cmap="gray")
        ax_img.plot(k_clicked[0], k_clicked[1], '+r', markersize=15)
        ax_img.plot([k_clicked[0]-delta,k_clicked[0]+delta],[k_clicked[1]-delta,k_clicked[1]-delta],'r')
        ax_img.plot([k_clicked[0]-delta,k_clicked[0]-delta],[k_clicked[1]-delta,k_clicked[1]+delta],'r')
        ax_img.plot([k_clicked[0]-delta,k_clicked[0]+delta],[k_clicked[1]+delta,k_clicked[1]+delta],'r')
        ax_img.plot([k_clicked[0]+delta,k_clicked[0]+delta],[k_clicked[1]+delta,k_clicked[1]-delta],'r')
        ax_img.plot(p_clicked[0], p_clicked[1], '+b', markersize=15)
        ax_img.plot([p_clicked[0]-delta,p_clicked[0]+delta],[p_clicked[1]-delta,p_clicked[1]-delta],'b')
        ax_img.plot([p_clicked[0]-delta,p_clicked[0]-delta],[p_clicked[1]-delta,p_clicked[1]+delta],'b')
        ax_img.plot([p_clicked[0]-delta,p_clicked[0]+delta],[p_clicked[1]+delta,p_clicked[1]+delta],'b')
        ax_img.plot([p_clicked[0]+delta,p_clicked[0]+delta],[p_clicked[1]+delta,p_clicked[1]-delta],'b')

        query = img_gray[p_clicked[1]-delta:p_clicked[1]+delta, p_clicked[0]-delta:p_clicked[0]+delta]
        query_big = cv.resize(query,(kernel_size*10,kernel_size*10), interpolation=cv.INTER_NEAREST)

        ax_query.imshow(query_big,cmap="gray")
            
        ssd = np.sum((kernel-query)**2)

        ncc = np.sum((kernel-np.mean(kernel))*(query-np.mean(query)))/(np.std(kernel)*np.std(query)*kernel.size)


        ax_query.set_title('SSD = ' + str(ssd) + '  NCC = ' + '{ncc_val:0.2f}'.format(ncc_val=ncc))
        plt.draw()  # We update the former image without create a new context


    plt.show()

if __name__ == '__main__':
    main()
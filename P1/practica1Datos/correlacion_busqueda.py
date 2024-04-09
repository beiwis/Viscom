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

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


def index2coords(index,nColumns):
    iRow = index // nColumns
    jColumn = index - iRow*nColumns
    return iRow, jColumn

def plot_box(ax_handle,x,y,delta):
    ax_handle.plot(x, y, '+r', markersize=15)
    ax_handle.plot([x-delta,x+delta],[y-delta,y-delta],'r')
    ax_handle.plot([x-delta,x-delta],[y-delta,y+delta],'r')
    ax_handle.plot([x-delta,x+delta],[y+delta,y+delta],'r')
    ax_handle.plot([x+delta,x+delta],[y+delta,y-delta],'r')    


def rotate_img(img, angle):
    """ Function to rotate an image [img] by [angle] degrees """
    center = (img.shape[1]/2, img.shape[0]/2)
    r_mat = cv.getRotationMatrix2D(center, angle, 1.0)
    rotated_img = cv.warpAffine(img, r_mat,
    (img.shape[1],img.shape[0]))
    return rotated_img


def main():

    img = cv.imread('board.jpg')
    img = cv.cvtColor(img,cv.COLOR_BGR2RGB) # Load image 1 and conversion from BGR 2 RGB    

    kernel_size = 21
    delta = int((kernel_size-1)/2)

    plt.figure(1)
    plt.title('Input image')    
    plt.imshow(img)
    coord_clicked_point = plt.ginput(1, show_clicks=False)
    k_clicked = np.array([coord_clicked_point[0][0], coord_clicked_point[0][1]])
    k_clicked = np.round(k_clicked).astype(np.uint16)
    plot_box(plt,k_clicked[0],k_clicked[1],delta)
    plt.draw() 
    kernel = img[k_clicked[1]-delta:k_clicked[1]+delta, k_clicked[0]-delta:k_clicked[0]+delta]
    kernel_big = cv.resize(kernel,(kernel_size*10,kernel_size*10), interpolation=cv.INTER_NEAREST)

    plt.figure(2)
    plt.title('Kernel')   
    plt.imshow(kernel_big)   
    plt.draw()  # We update the former image without create a new context
    plt.waitforbuttonpress()
    
    #Busqueda del pixel con mayor correlación
    method = cv.TM_CCOEFF_NORMED
    resOriginal = cv.matchTemplate(img,kernel,method) 
    sort_index = np.argsort(resOriginal.flatten())
    i1 = sort_index[-1]
    [mRowsCorr,nColumnsCorr] = resOriginal.shape[:2]
    [iRow1,jColumn1] = index2coords(i1,nColumnsCorr)

    plt.figure(3)
    plt.title('Target image NCC = ' + '{ncc_val:0.2f}'.format(ncc_val=resOriginal[iRow1,jColumn1]))    
    plt.imshow(img)
    plot_box(plt,jColumn1+10,iRow1+delta,delta)
    plt.draw()
    plt.waitforbuttonpress()    

    rotated_img = rotate_img(img, 65)
    plt.figure(4)
    plt.title('Rotated image')    
    plt.imshow(rotated_img)
    resOriginal = cv.matchTemplate(rotated_img,kernel,method) 
    sort_index = np.argsort(resOriginal.flatten())
    i1 = sort_index[-1]
    [mRowsCorr,nColumnsCorr] = resOriginal.shape[:2]
    [iRow1,jColumn1] = index2coords(i1,nColumnsCorr)

    plt.figure(5)
    plt.title('Target rotated image NCC = ' + '{ncc_val:0.2f}'.format(ncc_val=resOriginal[iRow1,jColumn1]))    
    plt.imshow(rotated_img)
    plot_box(plt,jColumn1+10,iRow1+delta,delta)
    plt.show()


if __name__ == '__main__':
    main()

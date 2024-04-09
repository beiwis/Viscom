import sys
from time import sleep
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from practica1Datos.convolucion import convolve

def main():
    plt.close('all')
    img = cv.imread('practica1Datos/stanfordPersRoom.png')
    imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    plt.imshow(imgGray, cmap ='gray')
    plt.title('Imagen 0. Imagen Grayscale.')
    plt.draw()
    plt.waitforbuttonpress()
    kernel_size = 31
    gaussian_kernel = cv.getGaussianKernel(kernel_size,0)
    gaussian_kernel_2d = np.outer(gaussian_kernel, gaussian_kernel.T)
    gaussian_kernel_2d_big = cv.resize(gaussian_kernel_2d*255,(kernel_size*10,kernel_size*10), interpolation= cv.INTER_NEAREST)
    plt.imshow(gaussian_kernel_2d_big, cmap ='gray')
    plt.title('Imagen 1. Filtro kernel gausiano.')
    plt.draw()
    plt.waitforbuttonpress()
    #Option 1:
    img_convolved = cv.filter2D(imgGray, -1, gaussian_kernel_2d)
    plt.imshow(img_convolved, cmap ='gray')
    plt.title('Imagen 2. Convolucion CV.')
    plt.draw()
    plt.waitforbuttonpress()
    #Option 2:
    #img_convolved = convolve(imgGray, gaussian_kernel_2d)
    plt.imshow(img_convolved, cmap ='gray')
    plt.title('Imagen 3. Convolucion manual.')
    plt.draw()
    plt.waitforbuttonpress()
    

    ############# FILTRO DE SOBEL:
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    img_sobel_x = cv.filter2D(imgGray, -1, sobel_x)
    plt.imshow(img_sobel_x, cmap ='gray')
    plt.title('Imagen 4.1. Filtro de Sobel x.')
    plt.draw()
    plt.waitforbuttonpress()
    img_sobel_y = cv.filter2D(imgGray, -1, sobel_y)
    plt.imshow(img_sobel_y, cmap ='gray')
    plt.title('Imagen 4.2. Filtro de Sobel y.')
    plt.draw()
    plt.waitforbuttonpress()
    img_sobel = cv.addWeighted(img_sobel_x, 0.5, img_sobel_y, 0.5,0)
    plt.imshow(img_sobel, cmap ='gray')
    plt.title('Imagen 4.3. Filtro de Sobel.')
    plt.draw()
    plt.waitforbuttonpress()
    
    
    ############# FILTRO DE CANNY
    low_threshold = 50
    ratio = 3
    kernel_size = 5
    img_canny = cv.Canny(imgGray, low_threshold, low_threshold*ratio,
    kernel_size)
    plt.imshow(img_canny, cmap ='gray')
    plt.title('Imagen 5. Filtro de Canny.')
    plt.draw()
    plt.waitforbuttonpress()


    ############# TRANSFORMACIONES GEOMETRICAS
    def rotate_img(img, angle):
        """ Function to rotate an image [img] by [angle] degrees """
        center = (img.shape[1]/2, img.shape[0]/2)
        r_mat = cv.getRotationMatrix2D(center, angle, 1.0)
        rotated_img = cv.warpAffine(img, r_mat,
        (img.shape[1],img.shape[0]))
        return rotated_img
    
    img_rotated = rotate_img(img, 45)
    plt.imshow(img_rotated, cmap ='gray')
    plt.title('Imagen 6. Transformación geométrica.')
    plt.show()
    # plt.draw()
    # plt.waitforbuttonpress()



    ## Apartado algo
    """ assert img is not None, "file could not be read, check with os.path.exists()"
    ret,thresh1 = cv.threshold(img,127,255,cv.THRESH_BINARY)
    ret,thresh2 = cv.threshold(img,127,255,cv.THRESH_BINARY_INV)
    ret,thresh3 = cv.threshold(img,127,255,cv.THRESH_TRUNC)
    ret,thresh4 = cv.threshold(img,127,255,cv.THRESH_TOZERO)
    ret,thresh5 = cv.threshold(img,127,255,cv.THRESH_TOZERO_INV)
    titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
    images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]
    for i in range(6):
        plt.subplot(2,3,i+1),plt.imshow(images[i],'gray',vmin=0,vmax=255)
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])
    plt.show() """
    

if __name__ == '__main__':
    main()
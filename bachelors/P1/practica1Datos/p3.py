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

    img = cv.imread('stanfordPersRoom.png')

    img = cv.cvtColor(img,cv.COLOR_BGR2RGB) # Load image 1 and conversion from BGR 2 RGB

    figure_1_id = 1
    plt.figure(figure_1_id)
    plt.imshow(img)
    plt.title('Imagen 1. Click para continuar.')
    plt.draw()
    plt.waitforbuttonpress()

     # Add the point "p_a" to the figure
    x_A = 200
    y_A = 300
    plt.plot(x_A, y_A, '+r', markersize=15)
    plt.title('Imagen 1 con punto A = (' + str(x_A) + ',' + str(y_A) + '). Click para continuar.')
    plt.draw()  # We update the figure display
    plt.waitforbuttonpress()

    # You can select the RGB color with values in [0, 1]
    r = 1
    g = 0
    b = 0
    color_red = (r, g, b)
    texto = 'Punto A'
    plt.text(x_A+20, y_A+20, texto, fontsize=15, color=color_red)
    plt.title('Imagen 1 con punto A = (' + str(x_A) + ',' + str(y_A) + ') y texto. Click para continuar.')
    plt.draw()  # We update the figure display
    plt.waitforbuttonpress()
    plt.close()

    # Picking image point coordinates in an image
    figure_2_id = 2
    plt.figure(figure_2_id)
    plt.imshow(img)
    plt.title('Imagen 1 - Haz click para seleccionar punto')
    coord_clicked_point = plt.ginput(1, show_clicks=False)
    p_clicked = np.array([coord_clicked_point[0][0], coord_clicked_point[0][1]])
    # We draw the point with its label
    plt.plot(p_clicked[0], p_clicked[1], '+r', markersize=15)
    plt.text(p_clicked[0]+20, p_clicked[1]+20,'Punto seleccionado',
             fontsize=15, color='r')
    plt.draw()  # We update the former image without create a new context
    plt.waitforbuttonpress()
    plt.close()

    # Drawing lines
    figure_3_id = 3
    plt.figure(figure_3_id)
    plt.imshow(img)
    plt.title('Haz click para seleccionar el origen de la recta')
    coord_clicked_point = plt.ginput(1, show_clicks=False)
    o_clicked = np.array([coord_clicked_point[0][0], coord_clicked_point[0][1]])
    # We draw the point with its label
    plt.plot(o_clicked[0], o_clicked[1], '+r', markersize=15)
    plt.draw()  # We update the former image without create a new context

    plt.title('Haz click para seleccionar el final de la recta')
    coord_clicked_point = plt.ginput(1, show_clicks=False)
    x_clicked = np.array([coord_clicked_point[0][0], coord_clicked_point[0][1]])
    # We draw the point with its label
    plt.plot([o_clicked[0],x_clicked[0]], [o_clicked[1],x_clicked[1]], 'r')
    plt.plot(x_clicked[0], x_clicked[1], '+r', markersize=15)
    # For more information about text formating visit: https://www.w3schools.com/python/python_string_formatting.asp
    plt.show() # ¿Cual es la diferencia entre draw() y show()?



#Esto sirve para ejecutar el codigo que haya despues en vez de ejecutar todo el codigo
if __name__ == '__main__':
    main()
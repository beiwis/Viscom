# Importamos las librerias que vayamos a utilizar
import sys
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def main():
    i = 0
    print("Hello world from Python!")
    # sys.argv es la lista que contiene todos los parámetros de la línea de comando
    # el parámetro en la posición 0 es el nombre del programa
    # el siguiente bucle imprime todos los argumentos por pantalla
    for param in sys.argv:
        print("El parametro {} es : {}".format(i,param))
        i+=1

if __name__ == '__main__':
    main()
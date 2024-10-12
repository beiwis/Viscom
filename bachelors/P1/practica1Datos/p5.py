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

import cv2 as cv

def main():
    cap = cv.VideoCapture('.\sec_test.mp4')
    tickFreq = cv.getTickFrequency()
    fps = cap.get(cv.CAP_PROP_FPS)
    if fps==0:
        print('Could not retrieve FPS from video')
        exit()
    else:
        tDelta = 1/fps
    width  = cap.get(cv.CAP_PROP_FRAME_WIDTH)   # float `width`
    height = cap.get(cv.CAP_PROP_FRAME_HEIGHT) 

    if not cap.isOpened():
        print("File not found")
        exit()

    output_file = 'sec_test_canny.mp4'
    fourcc = cv.VideoWriter_fourcc(*'mp4v')  # You can change the codec as needed
    fps_save = fps
    output_video = cv.VideoWriter(output_file, -1, fps_save, (int(width),int(height)))
    while cap.isOpened():
        tic = cv.getTickCount()
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        cv.imshow('frame', frame)
        output_frame = cannyFilter(frame)
        output_video.write(output_frame)

        toc = cv.getTickCount()
        time_taken = (toc-tic)/tickFreq

        delayInMilliSeconds = (tDelta-time_taken)*1000

        if delayInMilliSeconds>0:
            k = cv.waitKey(int(delayInMilliSeconds))
        else:
            k = cv.waitKey(1)
            print('Not in real-time')

        if k==ord('s'):
            break

    cap.release()
    output_video.release()

    cv.destroyAllWindows()


    return 0

def cannyFilter(img):
    imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    low_threshold = 50
    ratio = 3
    kernel_size = 5
    img_canny = cv.Canny(imgGray, low_threshold, low_threshold*ratio,
    kernel_size)
    return img_canny

#Esto sirve para ejecutar el codigo que haya despues en vez de ejecutar todo el codigo
if __name__ == '__main__':
    main()
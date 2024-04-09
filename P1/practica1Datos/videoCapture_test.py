import cv2 as cv
cap = cv.VideoCapture('./sec_test.mp4')

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break
        fps = cap.get(cv.CAP_PROP_FPS)
        print("FPS:", fps)
    cv.imshow('Video', frame)

    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
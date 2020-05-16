import cv2
import numpy as np

cap = cv2.VideoCapture(2)

while(cap.isOpened()):
    ret, frame = cap.read()
    frame = cv2.resize(frame, (150,150))
    cv2.imshow('Cam test', frame)
    print(frame.size)
    break

    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

cap.release()
cv2.destroyAllWindows()

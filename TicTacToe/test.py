import cv2
cap = cv2.VideoCapture(0)


while cap.isOpened():
    if cv2.waitKey(5) & 0xFF == 27:
        break
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue
   
    image = cv2.flip(image, 1)
    cv2.imshow('MediaPipe Hands', image)

cap.release()
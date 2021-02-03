import cv2, time

cap = cv2.VideoCapture("rtsp://admin:laser123@192.168.1.64:554/Streaming/Channels/101/")

time.sleep(2)

while (True):

    ret, frame = cap.read()
    if ret == 1:
        frame = cv2.flip(frame, 0)
        cv2.imshow('frame', frame)
    else:
        print("no video")
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
import cv2, time
import ptz_control

# callback함수
def mouse_move(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        ptz.move_relative(-(640-x)/6400, -(360-y)/3600, 0)

cap = cv2.VideoCapture("rtsp://admin:laser123@192.168.1.64:554/Streaming/Channels/101/")

# time.sleep(2)
ptz = ptz_control.ptzControl()
cv2.namedWindow("frame")
cv2.setMouseCallback('frame', mouse_move)

while (True):
    key = cv2.waitKey(1) & 0xFF
    ret, frame = cap.read()
    if ret == 1:
        frame = cv2.flip(frame, 0)
        cv2.imshow('frame', frame)
    else:
        print("no video")
        break
    if key == ord('q'):
        break

    elif key == ord("w"):
        ptz.move_relative(0, -0.1, 0)

    elif key == ord("s"):
        ptz.move_relative(0, 0.1, 0)

    elif key == ord("a"):
        ptz.move_relative(-0.1, 0, 0)

    elif key == ord("d"):
        ptz.move_relative(0.1, 0, 0)

    elif key == ord("="):
        ptz.zoom_relative(0.1, 0)

    elif key == ord("-"):
        ptz.zoom_relative(-0.1, 0)

cap.release()
cv2.destroyAllWindows()
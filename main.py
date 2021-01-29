from time import sleep
from onvif import ONVIFCamera
import cv2, time

cap = cv2.VideoCapture("rtsp://admin:admin@192.168.1.30:554/11")

time.sleep(2)

while (True):
    ret, frame = cap.read()
    print(ret)
    if ret == 1:
        cv2.imshow('frame', frame)
    else:
        print("no video")
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

XMAX = 1
XMIN = -1
YMAX = 1
YMIN = -1


def perform_move(ptz, request, timeout):
    # Start continuous move
    ptz.ContinuousMove(request)
    # Wait a certain time
    sleep(timeout)
    request.PanTilt = 1
    # Stop continuous move
    ptz.Stop(request)  # {'ProfileToken': request.ProfileToken})


def move_up(ptz, request, timeout=2):
    print('move up...')
    request.Velocity.PanTilt._x = 0
    request.Velocity.PanTilt._y = YMAX
    perform_move(ptz, request, timeout)


def move_down(ptz, request, timeout=2):
    print('move down...')
    request.Velocity.PanTilt._x = 0
    request.Velocity.PanTilt._y = YMIN
    perform_move(ptz, request, timeout)


def move_right(ptz, request, timeout=2):
    print('move right...')
    request.Velocity.PanTilt._x = XMAX
    request.Velocity.PanTilt._y = 0
    perform_move(ptz, request, timeout)


def move_left(ptz, request, timeout=2):
    print('move left...')
    request.Velocity.PanTilt._x = XMIN
    request.Velocity.PanTilt._y = 0
    perform_move(ptz, request, timeout)


def continuous_move():
    mycam = ONVIFCamera('192.168.1.30', 8080, 'admin', 'admin')
    # Create media service object
    media = mycam.create_media_service()
    # Create ptz service object
    ptz = mycam.create_ptz_service()

    # Get target profile
    media_profile = media.GetProfiles()[0];
    print(media_profile)

    # Get PTZ configuration options for getting continuous move range
    request = ptz.create_type('GetConfigurationOptions')
    request.ConfigurationToken = media_profile.PTZConfiguration._token
    ptz_configuration_options = ptz.GetConfigurationOptions(request)

    request = ptz.create_type('ContinuousMove')
    request.ProfileToken = media_profile._token

    # ptz.Stop({'ProfileToken': media_profile._token})

    # Get range of pan and tilt
    # NOTE: X and Y are velocity vector
    global XMAX, XMIN, YMAX, YMIN
    XMAX = ptz_configuration_options.Spaces.ContinuousPanTiltVelocitySpace[0].XRange.Max
    XMIN = ptz_configuration_options.Spaces.ContinuousPanTiltVelocitySpace[0].XRange.Min
    YMAX = ptz_configuration_options.Spaces.ContinuousPanTiltVelocitySpace[0].YRange.Max
    YMIN = ptz_configuration_options.Spaces.ContinuousPanTiltVelocitySpace[0].YRange.Min

    # move right
    move_right(ptz, request)

    # move left
    move_left(ptz, request)

    # Move up
    move_up(ptz, request)

    # move down
    move_down(ptz, request)

if __name__ == '__main__':
    continuous_move()
    cap.release()
    cv2.destroyAllWindows()
# python object_tracking.py --video ../videos/Group_display.mp4

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
# from cv2 import dnn_superres
import pandas as pd
import ptz_control
import os, struct, array
from fcntl import ioctl
from threading import *

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", type=str, default="",
	help="path to (optional) output video file")
ap.add_argument("-t", "--tracker", type=str, default="csrt",
	help="OpenCV object tracker type")
ap.add_argument("-s", "--sres", type=str, default="True",
	help="Apply Super_res")
ap.add_argument("-d", "--dt", type=float, default=30,
	help="Apply Super_res")
args = vars(ap.parse_args())


# OPENCV_OBJECT_TRACKERS = {
# 	"csrt": cv2.TrackerCSRT_create,
# 	"kcf": cv2.TrackerKCF_create,
# 	"boosting": cv2.TrackerBoosting_create,
# 	"mil": cv2.TrackerMIL_create,
# 	"tld": cv2.TrackerTLD_create,
# 	"medianflow": cv2.TrackerMedianFlow_create,
# 	"mosse": cv2.TrackerMOSSE_create,
# }

# grab the appropriate object tracker using our dictionary of
# OpenCV object tracker objects
# tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()


# Iterate over the joystick devices.
print('Available devices:')

for fn in os.listdir('/dev/input'):
    if fn.startswith('js'):
        print('  /dev/input/%s' % (fn))

# We'll store the states here.
axis_states = {}
button_states = {}

# These constants were borrowed from linux/input.h
axis_names = {
    0x00 : 'x',
    0x01 : 'y',
    0x02 : 'z',
    0x03 : 'rx',
    0x04 : 'ry',
    0x05 : 'rz',
    0x06 : 'trottle',
    0x07 : 'rudder',
    0x08 : 'wheel',
    0x09 : 'gas',
    0x0a : 'brake',
    0x10 : 'hat0x',
    0x11 : 'hat0y',
    0x12 : 'hat1x',
    0x13 : 'hat1y',
    0x14 : 'hat2x',
    0x15 : 'hat2y',
    0x16 : 'hat3x',
    0x17 : 'hat3y',
    0x18 : 'pressure',
    0x19 : 'distance',
    0x1a : 'tilt_x',
    0x1b : 'tilt_y',
    0x1c : 'tool_width',
    0x20 : 'volume',
    0x28 : 'misc',
}

button_names = {
    0x120 : 'trigger',
    0x121 : 'thumb',
    0x122 : 'thumb2',
    0x123 : 'top',
    0x124 : 'top2',
    0x125 : 'pinkie',
    0x126 : 'base',
    0x127 : 'base2',
    0x128 : 'base3',
    0x129 : 'base4',
    0x12a : 'base5',
    0x12b : 'base6',
	0x12c : 'base7',
	0x12d : 'base8',
	0x12e : 'base9',
    0x12f : 'dead',
    0x130 : 'a',
    0x131 : 'b',
    0x132 : 'c',
    0x133 : 'x',
    0x134 : 'y',
    0x135 : 'z',
    0x136 : 'tl',
    0x137 : 'tr',
    0x138 : 'tl2',
    0x139 : 'tr2',
    0x13a : 'select',
    0x13b : 'start',
    0x13c : 'mode',
    0x13d : 'thumbl',
    0x13e : 'thumbr',

    0x220 : 'dpad_up',
    0x221 : 'dpad_down',
    0x222 : 'dpad_left',
    0x223 : 'dpad_right',

    # XBox 360 controller uses these codes.
    0x2c0 : 'dpad_left',
    0x2c1 : 'dpad_right',
    0x2c2 : 'dpad_up',
    0x2c3 : 'dpad_down',
}

axis_map = []
button_map = []

# Open the joystick device.
fn = '/dev/input/js0'
print('Opening %s...' % fn)
jsdev = open(fn, 'rb')

# Get the device name.
#buf = bytearray(63)
buf = array.array('B', [0] * 64)
ioctl(jsdev, 0x80006a13 + (0x10000 * len(buf)), buf) # JSIOCGNAME(len)
js_name = buf.tobytes().rstrip(b'\x00').decode('utf-8')
print('Device name: %s' % js_name)

# Get number of axes and buttons.
buf = array.array('B', [0])
ioctl(jsdev, 0x80016a11, buf) # JSIOCGAXES
num_axes = buf[0]

buf = array.array('B', [0])
ioctl(jsdev, 0x80016a12, buf) # JSIOCGBUTTONS
num_buttons = buf[0]

# Get the axis map.
buf = array.array('B', [0] * 0x40)
ioctl(jsdev, 0x80406a32, buf) # JSIOCGAXMAP

for axis in buf[:num_axes]:
    axis_name = axis_names.get(axis, 'unknown(0x%02x)' % axis)
    axis_map.append(axis_name)
    axis_states[axis_name] = 0.0

# Get the button map.
buf = array.array('H', [0] * 200)
ioctl(jsdev, 0x80406a34, buf) # JSIOCGBTNMAP

for btn in buf[:num_buttons]:
    btn_name = button_names.get(btn, 'unknown(0x%03x)' % btn)
    button_map.append(btn_name)
    button_states[btn_name] = 0

print('%d axes found: %s' % (num_axes, ', '.join(axis_map)))
print('%d buttons found: %s' % (num_buttons, ', '.join(button_map)))

# initialize the bounding box coordinates of the object we are going
# to track
initBB = None

vs = cv2.VideoCapture("rtsp://admin:laser123@192.168.1.64:554/Streaming/Channels/101/")
# initialize the FPS throughput estimator
fps = None
writer = None
data = []
count = 0

ptz = ptz_control.ptzControl()

def js(jsdev):
	js_x = 0
	js_y = 0
	while True:
		evbuf = jsdev.read(8)
		if evbuf:
			_, value, type, number = struct.unpack('IhBB', evbuf)

			if type & 0x80:
					print("(initial)", end="")

			if type & 0x01:
				button = button_map[number]
				if button:
					button_states[button] = value
					if value:
						print("%s pressed" % (button))
					else:
						print("%s released" % (button))

			if type & 0x02:
				axis = axis_map[number]
				if axis:
					fvalue = value / 32767.0
					axis_states[axis] = fvalue
					if axis == 'x':
						js_x = fvalue				
					elif axis == 'y':
						js_y = fvalue
					elif axis == 'hat0x' or axis == 'hat0y':
						ptz.zoom(fvalue)

					# print("%s: %.3f" % (axis, fvalue))
		ptz.move_continuous(js_x, js_y)

js_t = Thread(target=js, args=(jsdev,))
js_t.daemon = True
js_t.start()

# loop over frames from the video stream
while True:	
	# grab the current frame, then handle if we are using a
	# VideoStream or VideoCapture object
	success, frame = vs.read()
	# check to see if we have reached the end of the stream
	if frame is None:
		break
	frame = cv2.flip(frame, 0)
	# check to see if we are currently tracking an object
	if initBB is not None:
		# grab the new bounding box coordinates of the object
		(success, box) = tracker.update(frame)
		# check to see if the tracking was a success
		if success:
			(x, y, w, h) = [int(v) for v in box]
			cv2.rectangle(frame, (x, y), (x + w, y + h),
				(0, 255, 0), 1)
			if count % 30 == 0:
				move_x = -(640-x)/6400
				move_y = -(360-y)/3600
				velocity = np.sqrt(move_x**2 + move_y**2)
				if velocity > 0.001 and velocity <= 0.01:
					ptz.move_relative(move_x, move_y, velocity)
				if velocity > 0.01:
					ptz.move_relative(move_x, move_y, 1)
		# (gt_x, gt_y, gt_w, gt_h) = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)		
		# frame = cv2.rectangle(frame, (gt_x, gt_y), (gt_x + gt_w, gt_y + gt_h), (0, 0, 255), 1)
		# area_x = min(x + w, gt_x + gt_w) - max(x, gt_x)
		# area_y = min(y + h, gt_y + gt_h) - max(y, gt_y)
		# iou = area_x * area_y / (w * h + gt_w * gt_h - area_x * area_y)

		# data.append([count, x, y, w, h])
		
		# frame = cv2.resize(frame, dsize=(640, 480), interpolation=cv2.INTER_CUBIC)
		
		# update the FPS counter
		fps.update()
		fps.stop()

		# initialize the set of information we'll be displaying on
		# the frame
		info = [
			# ("Tracker", args["tracker"]),
			# ("counter: {}".format(count)),
			("FPS", "{:.2f}".format(fps.fps())),
			("{} {}".format(x, y), "{} {}".format(w, h)),
			# ("{} {}".format(gt_x, gt_y), "{} {}".format(gt_w, gt_h)),
		]

		for (i, (k, v)) in enumerate(info):
			text = "{}: {}".format(k, v)
			cv2.putText(frame, text, (10, 60 - ((i * 20) + 20)),
				cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

		if args["output"] != "" and writer is None:
		# initialize our video writer
			fourcc = cv2.VideoWriter_fourcc(*"mp4v")
			writer = cv2.VideoWriter(args["output"], fourcc, 30,
				(frame.shape[1], frame.shape[0]), True)	
		count += 1
	# if count == 1000:
	# 	break

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
		
	# if an output video file path has been supplied and the video
	# writer has not been initialized, do so now
	

	# if the video writer is not None, write the frame to the output
	# video file
	if writer is not None:
		writer.write(frame)

	# if the 's' key is selected, we are going to "select" a bounding
	# box to track
	if key == ord("c"):
		tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()
		# select the bounding box of the object we want to track (make
		# sure you press ENTER or SPACE after selecting the ROI)
		initBB = cv2.selectROI("Frame", frame, fromCenter=False,
			showCrosshair=True)
		# start OpenCV object tracker using the supplied bounding box
		# coordinates, then start the FPS throughput estimator as well
		tracker.init(frame, initBB)

		fps = FPS().start()

	elif key == ord("w"):
		ptz.move_relative(0, -0.05, 0)

	elif key == ord("s"):
		ptz.move_relative(0, 0.05, 0)

	elif key == ord("a"):
		ptz.move_relative(-0.05, 0, 0)

	elif key == ord("d"):
		ptz.move_relative(0.05, 0, 0)

	elif key == ord("="):
		ptz.zoom_relative(0.1, 0)

	elif key == ord("-"):
		ptz.zoom_relative(-0.1, 0)

	# if the `q` key was pressed, break from the loop
	elif key == ord("q"):
		break

vs.release()
# close all windows
# df = pd.DataFrame(data)
# df.to_csv('output.csv', index = False)
cv2.destroyAllWindows()

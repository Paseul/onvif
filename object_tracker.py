# python object_tracking.py --video ../videos/Group_display.mp4

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
import os
from cv2 import dnn_superres
import pandas as pd
import ptz_control

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


OPENCV_OBJECT_TRACKERS = {
	"csrt": cv2.TrackerCSRT_create,
	"kcf": cv2.TrackerKCF_create,
	"boosting": cv2.TrackerBoosting_create,
	"mil": cv2.TrackerMIL_create,
	"tld": cv2.TrackerTLD_create,
	"medianflow": cv2.TrackerMedianFlow_create,
	"mosse": cv2.TrackerMOSSE_create,
}

# grab the appropriate object tracker using our dictionary of
# OpenCV object tracker objects
tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()

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
			if count % 10 == 0:
				move_x = -(640-x)/6400
				move_y = -(360-y)/3600
				velocity = np.sqrt(move_x**2 + move_y**2)
				if velocity > 0.001 and velocity <= 0.01:
					ptz.move_relative(move_x, move_y, velocity)
				if velocity > 0.01:
					print(velocity)
					ptz.move_relative(move_x, move_y, 100)
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
	elif key == ord("w"):
		ptz.move_relative(0, -0.01, 0)

	elif key == ord("s"):
		ptz.move_relative(0, 0.01, 0)

	elif key == ord("a"):
		ptz.move_relative(-0.01, 0, 0)

	elif key == ord("d"):
		ptz.move_relative(0.01, 0, 0)

	elif key == ord("="):
		ptz.zoom_relative(0.1, 0)

	elif key == ord("-"):
		ptz.zoom_relative(-0.1, 0)

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

	# if the `q` key was pressed, break from the loop
	elif key == ord("q"):
		break

vs.release()
# close all windows
# df = pd.DataFrame(data)
# df.to_csv('output.csv', index = False)
cv2.destroyAllWindows()

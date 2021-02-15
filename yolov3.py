import cv2
from imutils.video import FPS
import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", type=str, default="yolov3_out.mp4",
	help="path to (optional) output video file")
ap.add_argument("-u", "--gpu", type=bool, default=0,
	help="boolean indicating if CUDA GPU should be used")
args = vars(ap.parse_args())

CONFIDENCE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.4
COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]

labelsPath = "yolo-coco/coco.names"
LABELS = open(labelsPath).read().strip().split("\n")

vc = cv2.VideoCapture("rtsp://admin:laser123@192.168.1.64:554/Streaming/Channels/101/")

# derive the paths to the YOLO weights and model configuration
weightsPath = "yolo-coco/yolov3.weights"
configPath = "yolo-coco/yolov3.cfg"

# load our YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# check if we are going to use GPU
if args["gpu"]:
    # set CUDA as the preferable backend and target
    print("[INFO] setting preferable backend and target to CUDA...")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255)

fps = None
writer = None
count = 0
fps = FPS().start()
while cv2.waitKey(1) < 1:
    (grabbed, frame) = vc.read()
    if not grabbed:
        exit()
    frame = cv2.flip(frame, 0)
    if count % 5 == 0:
        classes, scores, boxes = model.detect(frame, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

        for (classid, score, box) in zip(classes, scores, boxes):
            color = COLORS[int(classid) % len(COLORS)]
            label = "%s : %f" % (LABELS[classid[0]], score)
            cv2.rectangle(frame, box, color, 2)
            cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    fps.update()
    fps.stop()

    fps_label = "FPS: {:.2f}".format(fps.fps())
    cv2.putText(frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.imshow("detections", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("s") and args["output"] != "" and writer is None:
        # initialize our video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args["output"], fourcc, 30,
            (frame.shape[1], frame.shape[0]), True)

    # if the `q` key was pressed, break from the loop
    elif key == ord("q"):
        break

    # if the video writer is not None, write the frame to the output
    # video file
    if writer is not None:
        writer.write(frame)
    count += 1
cv2.destroyAllWindows()
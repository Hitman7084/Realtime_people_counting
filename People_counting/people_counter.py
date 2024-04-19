from tracker.centroidtracker import CentroidTracker
from tracker.trackableobject import TrackableObject
from imutils.video import VideoStream
from itertools import zip_longest
from utils.mailer import Mailer
from imutils.video import FPS
from utils import thread
import numpy as np
import threading
import argparse
import datetime
import schedule
import logging
import imutils
import time
import dlib
import json
import csv
import cv2

start_time = time.time()

logging.basicConfig(level = logging.INFO, format = "[INFO] %(message)s")
logger = logging.getLogger(__name__)

with open("utils/config.json", "r") as file:
    config = json.load(file)

def parse_arguments():

    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--prototxt", required=False,
        help="path to Caffe 'deploy' prototxt file")
    ap.add_argument("-m", "--model", required=True,
        help="path to Caffe pre-trained model")
    ap.add_argument("-i", "--input", type=str,
        help="path to optional input video file")
    ap.add_argument("-o", "--output", type=str,
        help="path to optional output video file")
    # confidence default 0.4
    ap.add_argument("-c", "--confidence", type=float, default=0.4,
        help="minimum probability to filter weak detections")
    ap.add_argument("-s", "--skip-frames", type=int, default=30,
        help="# of skip frames between detections")
    args = vars(ap.parse_args())
    return args

def send_mail():
	Mailer().send(config["Email_Receive"])

def log_data(move_in, in_time, move_out, out_time):
	data = [move_in, in_time, move_out, out_time]
	export_data = zip_longest(*data, fillvalue = '')

	with open('utils/data/logs/counting_data.csv', 'w', newline = '') as myfile:
		wr = csv.writer(myfile, quoting = csv.QUOTE_ALL)
		if myfile.tell() == 0: # check if header rows are already existing
			wr.writerow(("Move In", "In Time", "Move Out", "Out Time"))
			wr.writerows(export_data)

def people_counter():
	# main function for people_counter.py
	args = parse_arguments()
	CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
		"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
		"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
		"sofa", "train", "tvmonitor"]

	# load our serialized model from disk
	net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

	# if a video path was not supplied, grab a reference to the ip camera
	if not args.get("input", False):
		logger.info("Starting the live stream..")
		vs = VideoStream(config["url"]).start()
		time.sleep(2.0)

	else:
		logger.info("Starting the video..")
		vs = cv2.VideoCapture(args["input"])

	writer = None

	W = None
	H = None

	ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
	trackers = []
	trackableObjects = {}

	totalFrames = 0
	totalDown = 0
	totalUp = 0

	total = []
	move_out = []
	move_in =[]
	out_time = []
	in_time = []

	fps = FPS().start()

	if config["Thread"]:
		vs = thread.ThreadingClass(config["url"])

	while True:
		frame = vs.read()
		frame = frame[1] if args.get("input", False) else frame

		if args["input"] is not None and frame is None:
			break

		frame = imutils.resize(frame, width = 500)
		rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

		# if the frame dimensions are empty, set them
		if W is None or H is None:
			(H, W) = frame.shape[:2]

		if args["output"] is not None and writer is None:
			fourcc = cv2.VideoWriter_fourcc(*"mp4v")
			writer = cv2.VideoWriter(args["output"], fourcc, 30,
				(W, H), True)

		status = "Waiting"
		rects = []

		if totalFrames % args["skip_frames"] == 0:
			status = "Detecting"
			trackers = []

			blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
			net.setInput(blob)
			detections = net.forward()

			for i in np.arange(0, detections.shape[2]):

				confidence = detections[0, 0, i, 2]

				if confidence > args["confidence"]:
					idx = int(detections[0, 0, i, 1])

					# if the class label is not a person, ignore it
					if CLASSES[idx] != "person":
						continue

					box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
					(startX, startY, endX, endY) = box.astype("int")

					tracker = dlib.correlation_tracker()
					rect = dlib.rectangle(startX, startY, endX, endY)
					tracker.start_track(rgb, rect)

					trackers.append(tracker)

		else:
			# loop over the trackers
			for tracker in trackers:
				status = "Tracking"

				tracker.update(rgb)
				pos = tracker.get_position()

				# unpack the position object
				startX = int(pos.left())
				startY = int(pos.top())
				endX = int(pos.right())
				endY = int(pos.bottom())

				# add the bounding box coordinates to the rectangles list
				rects.append((startX, startY, endX, endY))

		cv2.line(frame, (0, H // 2), (W, H // 2), (0, 0, 0), 3)
		cv2.putText(frame, "-Prediction border - Entrance-", (10, H - ((i * 20) + 200)),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

		objects = ct.update(rects)

		# loop over the tracked objects
		for (objectID, centroid) in objects.items():

			to = trackableObjects.get(objectID, None)

			# if there is no existing trackable object, create one
			if to is None:
				to = TrackableObject(objectID, centroid)

			# otherwise, there is a trackable object so we can utilize it
			# to determine direction
			else:
				y = [c[1] for c in to.centroids]
				direction = centroid[1] - np.mean(y)
				to.centroids.append(centroid)

				# check to see if the object has been counted or not
				if not to.counted:

					if direction < 0 and centroid[1] < H // 2:
						totalUp += 1
						date_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
						move_out.append(totalUp)
						out_time.append(date_time)
						to.counted = True


					elif direction > 0 and centroid[1] > H // 2:
						totalDown += 1
						date_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
						move_in.append(totalDown)
						in_time.append(date_time)
						# if the people limit exceeds over threshold, send an email alert
						if sum(total) >= config["Threshold"]:
							cv2.putText(frame, "-ALERT: People limit exceeded-", (10, frame.shape[0] - 80),
								cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 2)
							if config["ALERT"]:
								logger.info("Sending email alert..")
								email_thread = threading.Thread(target = send_mail)
								email_thread.daemon = True
								email_thread.start()
								logger.info("Alert sent!")
						to.counted = True
						# compute the sum of total people inside
						total = []
						total.append(len(move_in) - len(move_out))

			# store the trackable object in our dictionary
			trackableObjects[objectID] = to

			text = "ID {}".format(objectID)
			cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
			cv2.circle(frame, (centroid[0], centroid[1]), 4, (255, 255, 255), -1)

		# construct a tuple of information we will be displaying on the frame
		info_status = [
		("Exit", totalUp),
		("Enter", totalDown),
		("Status", status),
		]

		info_total = [
		("Total people inside", ', '.join(map(str, total))),
		]

		# display the output
		for (i, (k, v)) in enumerate(info_status):
			text = "{}: {}".format(k, v)
			cv2.putText(frame, text, (10, H - ((i * 20) + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

		for (i, (k, v)) in enumerate(info_total):
			text = "{}: {}".format(k, v)
			cv2.putText(frame, text, (265, H - ((i * 20) + 60)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

		# initiate a simple log to save the counting data
		if config["Log"]:
			log_data(move_in, in_time, move_out, out_time)

		# check to see if we should write the frame to disk
		if writer is not None:
			writer.write(frame)

		# show the output frame
		cv2.imshow("Real-Time Monitoring/Analysis Window", frame)
		key = cv2.waitKey(1) & 0xFF
		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break

		totalFrames += 1
		fps.update()

		# initiate the timer
		if config["Timer"]:
			# automatic timer to stop the live stream (set to 8 hours/28800s)
			end_time = time.time()
			num_seconds = (end_time - start_time)
			if num_seconds > 28800:
				break

	# stop the timer and display FPS information
	fps.stop()
	logger.info("Elapsed time: {:.2f}".format(fps.elapsed()))
	logger.info("Approx. FPS: {:.2f}".format(fps.fps()))

	# release the camera device/resource (issue 15)
	if config["Thread"]:
		vs.release()

	# close any open windows
	cv2.destroyAllWindows()

# initiate the scheduler
if config["Scheduler"]:

	schedule.every().day.at("09:00").do(people_counter)
	while True:
		schedule.run_pending()
else:
	people_counter()

from lib import config, thread
from lib.mailer import Mailer
from lib.detection import detect_people
from imutils.video import VideoStream, FPS
from scipy.spatial import distance as dist
import numpy as np
import argparse, imutils, cv2, os, time
import math
#, schedule

#----------------------------Parse req. arguments------------------------------#
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--input", type=str, default="",
# 	help="path to (optional) input video file")
# ap.add_argument("-o", "--output", type=str, default="",
# 	help="path to (optional) output video file")
# ap.add_argument("-d", "--display", type=int, default=1,
# 	help="whether or not output frame should be displayed")
# args = vars(ap.parse_args())
# #------------------------------------------------------------------------------#

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([config.MODEL_PATH, "labels.txt"])
LABELS = open(labelsPath).read().strip().split("\n")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([config.MODEL_PATH, "custom-yolov4-tiny-detector_best_person1.weights"])
configPath = os.path.sep.join([config.MODEL_PATH, "custom-yolov4-tiny-detector_person.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# check if we are going to use GPU
#if config.USE_GPU:
	# set CUDA as the preferable backend and target
	#print("")
	#print("[INFO] Looking for GPU")
	#net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
	#net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
file = "test6"
input = "C:/Omkar.Uttarwar/Beverly Kitchen/Social-Distancing/Videos/"+file+".mp4"
output = "C:/Omkar.Uttarwar/Beverly Kitchen/Social-Distancing/Outputs/"+file+".avi"
#file = "test_ten"
# if a video path was not supplied, grab a reference to the camera
# if not args.get("input", False):
# 	print("[INFO] Starting the live stream..")
# 	vs = cv2.VideoCapture(config.url)
# 	if config.Thread:
# 			cap = thread.ThreadingClass(config.url)
# 	time.sleep(2.0)

# # otherwise, grab a reference to the video file
# else:
# 	print("[INFO] Starting the video..")
vs = cv2.VideoCapture(input)
# 	if config.Thread:
# 			cap = thread.ThreadingClass(args["input"])

writer = None
# start the FPS counter
#fps = FPS().start()
f = open("C:/Omkar.Uttarwar/Beverly Kitchen/Social-Distancing/Reports/Report {}.txt".format(file), "w")
f.write("\n REPORT \n")


TotalFrames = 0
# loop over the frames from the video stream
TotalViolations = 0
TotalViolations5 = 0

soc_dis = {}
while True:
	# read the next frame from the file
	if config.Thread:
		frame = cap.read()

	else:
		(grabbed, frame) = vs.read()
		# if the frame was not grabbed, then we have reached the end of the stream
		if not grabbed:
			break
	
	# resize the frame and then detect people (and only people) in it
	#frame = imutils.resize(frame, width=700)
	results = detect_people(frame, net, ln,
		personIdx=[LABELS.index("Person") ])
		#, LABELS.index("PPE Kit")])

	# initialize the set of indexes that violate the max/min social distance limits
	serious = set()
	abnormal = set()

	# ensure there are *at least* two people detections (required in
	# order to compute our pairwise distance maps)
	#fourcc = cv2.VideoWriter_fourcc(*"MJPG")
	#writer = cv2.VideoWriter(output, fourcc, 25,
		#(frame.shape[1], frame.shape[0]), True)
	if len(results) >= 2:
		# extract all centroids from the results and compute the
		# Euclidean distances between all pairs of the centroids
		centroids = np.array([r[2] for r in results])
		D = dist.cdist(centroids, centroids, metric="euclidean")

		# loop over the upper triangular of the distance matrix
		for i in range(0, D.shape[0]):
			for j in range(i + 1, D.shape[1]):
				# check to see if the distance between any two
				# centroid pairs is less than the configured number of pixels
				if D[i, j] < config.MIN_DISTANCE:
					# update our violation set with the indexes of the centroid pairs
					serious.add(i)
					serious.add(j)
					#print(i,j)
                # update our abnormal set if the centroid distance is below max distance limit
				if (D[i, j] < config.MAX_DISTANCE) and not serious:
					abnormal.add(i)
					abnormal.add(j)

	# loop over the results
	for (i, (prob, bbox, centroid)) in enumerate(results):
		# extract the bounding box and centroid coordinates, then
		# initialize the color of the annotation
		(startX, startY, endX, endY) = bbox
		(cX, cY) = centroid
		color = (0, 255, 0)

		# if the index pair exists within the violation/abnormal sets, then update the color
		if i in serious:
			color = (0, 0, 255)
		elif i in abnormal:
			color = (0, 255, 255) #orange = (0, 165, 255)

		# draw (1) a bounding box around the person and (2) the
		# centroid coordinates of the person,
		#cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
		#cv2.circle(frame, (cX, cY), 5, color, 2)

	# draw some of the parameters
	Safe_Distance = "Safe distance: >{} px".format(config.MAX_DISTANCE)
	#cv2.putText(frame, Safe_Distance, (470, frame.shape[0] - 25),
	#	cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 0, 0), 2)
	Threshold = "Threshold limit: {}".format(config.Threshold)
	#cv2.putText(frame, Threshold, (470, frame.shape[0] - 50),
	#	cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 0, 0), 2)

    # draw the total number of social distancing violations on the output frame
	TotalViolations += len(serious)
	TotalViolations5 += len(serious)
	text = "Social Distancing violations: {}".format(len(serious))
	cv2.putText(frame, text, (10, frame.shape[0] - 55),
		cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 0, 255), 2)

	text1 = "Total violations: {}".format(TotalViolations)
	cv2.putText(frame, text1, (10, frame.shape[0] - 25),
		cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 0, 255), 2)
	#if len(serious) > 0:
		#print(D)
		#print(serious)
	#if len(serious)>0:

	#	soc_dis[TotalFrames] = 1

	#else:

	#	soc_dis[TotalFrames] = 0 

	if TotalFrames % 8 == 0:
		f.write("\nAt {} seconds:".format(TotalFrames/8))
		f.write("\nNumber of Social Distancing Violations: {}\n".format((math.ceil(TotalViolations5/8))))
		TotalViolations5 = 0
		#break
	# if TotalFrames % 40 == 0:
		
	# 	NotViolated = 0 in soc_dis.values()

	# 	if NotViolated:
	# 		soc_dis.clear()

	# 	else:
	# 		TotalViolations += 1
	# 		TotalViolations5 += 1
			#soc_dis.clear()
			#f.write("\nNumber of Social Distancing Violations: {}\n".format(TotalViolations5))
#------------------------------Alert function----------------------------------#
	# if len(serious) >= config.Threshold:
	# 	cv2.putText(frame, "-ALERT: Violations over limit-", (10, frame.shape[0] - 80),
	# 		cv2.FONT_HERSHEY_COMPLEX, 0.60, (0, 0, 255), 2)
	# 	if config.ALERT:
	# 		print("")
	# 		print('[INFO] Sending mail...')
	# 		Mailer().send(config.MAIL)
	# 		print('[INFO] Mail sent')
		#config.ALERT = False
#------------------------------------------------------------------------------#
	# check to see if the output frame should be displayed to our screen
	# if args["display"] > 0:
	# 	# show the output frame
	# 	cv2.imshow("Real-Time Monitoring/Analysis Window", frame)
	# 	key = cv2.waitKey(1) & 0xFF

	# 	# if the `q` key was pressed, break from the loop
	# 	if key == ord("q"):
	# 		break
    # update the FPS counter
	#fps.update()

	# if an output video file path has been supplied and the video
	TotalFrames += 1 
	if output != "" and writer is None:
		# initialize our video writer
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(output, fourcc, 8,
			(frame.shape[1], frame.shape[0]), True)

	# if the video writer is not None, write the frame to the output video file
	if writer is not None:
		writer.write(frame)

# stop the timer and display FPS information
#fps.stop()
#print("===========================")
#print("[INFO] Elasped time: {:.2f}".format(fps.elapsed()))
#print("[INFO] Approx. FPS: {:.2f}".format(fps.fps()))
f.close()
# close any open windows
cv2.destroyAllWindows()


#! usr/bin/python3.10
# import the necessary packages
from cv2 import FONT_HERSHEY_PLAIN
from imutils.video import VideoStream
from imutils.video import FPS
import face_recognition
import imutils
import pickle
import time
import cv2

#import RPi.GPIO as GPIO

#RELAY = INSERT PIN NUMBER HERE
#GPIO.setwarnings(False)
#GPIO.setmode(GPIO.BCM)
#GPIO.setup(RELAY, GPIO.OUT)
#GPIO.output(RELAY,GPIO.LOW)

# current name is set to unknown if the person is unknown
currentname = "unknown"
# encoding faces generated from the encode_faces.py, which is another saying for faces are trained.
encodings = "encodings.pickle"

# use the haar cascade

face_cascade = "haar_face.xml"
 
# face_cascade for face detection
print("[INFO] loading encodings + face detector…")
face_encodings = pickle.loads(open(encodings, "rb").read())
face_detector = cv2.CascadeClassifier(face_cascade)

# initialize the video stream and allow the camera sensor to warm up
print("starting video stream ... ")

video_show = VideoStream(src=0).start()

#video_show = VideoStream(usePiCamera=True).start()

time.sleep(2.0)

# start the FPS counter
fps = FPS().start()

# TO BE USED IN THE RASPBERRYPI CONNECTED TO THE LOCK

#prevTime = 0
#doorUnlock = False

#################

# while loop to serve as a "listening"
while True:
	# width can be resized to whatever, recommended is 400-800

	frame = video_show.read()
	frame = imutils.resize(frame, width=800)

	#convert input frame from blue,red,green to grayscale for the face detection
	#convert input frame from blue,red,green to red,green,blue for facial recognition
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

	# detect faces in the grayscale frame
	face_rectangle = face_detector.detectMultiScale(gray, 1.1, 4)

	# OpenCV returns bounding box coordinates in (x, y, w, h) order

	boxes = [(y, x + w, y + h, x) for (x, y, w, h) in face_rectangle]

	# compute the facial embeddings for each face bounding box
	encodings = face_recognition.face_encodings(rgb, boxes)
	names = []

	# loop over the facial embeddings
	for encoding in encodings:
		# attempt to match each face in the input image to our known
		# encodings
		matches = face_recognition.compare_faces(face_encodings["encodings"],
			encoding)
		name = "unknown person" #if face is not recognized, then print Unknown

		# check to see if we have found a match
		if True in matches:
			# find the indexes of all matched faces then initialize a
			# dictionary to count the total number of times each face
			# was matched
			matchedIdxs = [i for (i, b) in enumerate(matches) if b]
			counts = {}
			
			# to unlock the door
			#GPIO.output(RELAY,GPIO.HIGH)
			#prevTime = time.time()
			#doorUnlock = True
			#print("door unlock")
			

			# loop over the matched indexes and maintain a count for
			# each recognized face face
			for i in matchedIdxs:
				name = face_encodings["names"][i]
				counts[name] = counts.get(name, 0) + 1

			# determine the recognized face with the largest number
			# of votes (note: in the event of an unlikely tie Python
			# will select first entry in the dictionary)
			name = max(counts, key=counts.get)

			#If someone in your dataset is identified, print their name on the screen
			if currentname != name:
				currentname = name
				#print("Hello, ", name, ". Access granted.")

		# update the list of names
		names.append(name)
        
        #lock the door after 5 seconds
	#if doorUnlock == True and time.time() - prevTime > 5:
		#doorUnlock = False
		#GPIO.output(RELAY,GPIO.LOW)
		#print("door lock")

	# loop over the recognized faces
		for ((top, right, bottom, left), name) in zip(boxes, names):
		# draw rectangle on known face and unknown face
			cv2.rectangle(frame, (left, top), (right, bottom),
			(0, 255, 0), 1)
			y = top - 15 if top - 15 > 15 else top + 15
			cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
				.8, (0, 0, 0), 2)

	# display video
	cv2.imshow("Company X - Facial Recognition", frame)
	key = cv2.waitKey(1) & 0xFF

	# quit when 'q' key is pressed
	if key == ord("q"):
		break

	# update the FPS counter
	fps.update()

# stop fps counter
fps.stop()

# calculate fps and re-scale width on raspberry pi
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# destroy all
cv2.destroyAllWindows()
video_show.stop()
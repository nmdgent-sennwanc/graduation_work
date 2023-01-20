# Imports
import cv2
import tensorflow as tf
import numpy as np
import time
from sense_hat import SenseHat
from watchfiles import watch
from tensorflow.keras.models import load_model
 
# Loading in the classifier model
MODEL_PATH = r"/home/senne/Documenten/Programming/GraduationWork/ImageClassifier/imageclassifier.h5"
imageclassifier = load_model(MODEL_PATH)
 
# Declaring file path
FILE_PATH = r"/home/senne/Documenten/Programming/GraduationWork/FaceDetector/input_image/input_image.jpg"
 
# Classification
def run_classification():
	# Read in image
	img = cv2.imread(FILE_PATH)
	# Convert colors
	im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

	# Resize image to 256, 256
	resize = tf.image.resize(im_rgb, (256,256))

	# Do prediction via model
	yhat = imageclassifier.predict(np.expand_dims(resize/255, 0))
	print('Classification score of: ', yhat[0][0])
	# Because we are opening a door, we take a higher value
	if yhat > 0.80:
			# This is a member, door can be opened
			return True
	else:
			# This is an intruder, door cannot be opened
			return False

# Detecting changes
def detectChanges():
	# Declaring directory path			
	DIR_PATH = r"/home/senne/Documenten/Programming/GraduationWork/FaceDetector/input_image"
	print('Detecting changes has started at: ', DIR_PATH)

	for changes in watch(DIR_PATH):
		# Run classification with changed image
		classification_value = run_classification()

		# Open/Close door (send signal to sensehat)
		if classification_value == True:
			print('Person is MEMBER, door can be opened')
			sense.set_pixels(open_pixels)
			time.sleep(2)
			sense.clear()
		else: 
			print('Person is INTRUDER, door must be closed')
			sense.set_pixels(closed_pixels)
			time.sleep(2)
			sense.clear()

# Sense hat settings
sense = SenseHat()

# Define some colours
g = (0, 255, 0) # Green
r = (255, 0, 0) # Red

closed_pixels = [
	r, r, r, r, r, r, r, r,
	r, r, r, r, r, r, r, r,
	r, r, r, r, r, r, r, r,
	r, r, r, r, r, r, r, r,
	r, r, r, r, r, r, r, r,
	r, r, r, r, r, r, r, r,
	r, r, r, r, r, r, r, r,
	r, r, r, r, r, r, r, r
]

open_pixels = [
	g, g, g, g, g, g, g, g,
	g, g, g, g, g, g, g, g,
	g, g, g, g, g, g, g, g,
	g, g, g, g, g, g, g, g,
	g, g, g, g, g, g, g, g,
	g, g, g, g, g, g, g, g,
	g, g, g, g, g, g, g, g,
	g, g, g, g, g, g, g, g
]

# Trying to detect changes in input image directory
try: 
	# Clear sensehat
	sense.clear()
	print('Trying to detect changes...')

	# Run detection 
	detectChanges()
except KeyboardInterrupt:
	print('Stopped by KeyboardInterrupt')
else:
	print('No exceptions thrown')
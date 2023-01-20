import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

# Loading in the facetracking model
MODEL_PATH = r"/home/senne/Documenten/Programming/GraduationWork/FaceDetector/facetracker_v2.h5"
facetracker = load_model('facetracker_v2.h5')

# Function to save image from camera
def saveImage(capture, sample_coords, frame):
	SAVE_PATH =  os.path.join('input_image', 'input_image.jpg')
	x_min = (sample_coords[0]*450).astype(int) - 20
	y_min = (sample_coords[1]*450).astype(int) - 20
	x_max = (sample_coords[2]*450).astype(int) + 20
	y_max = (sample_coords[3]*450).astype(int) + 20

	h_array = [y_min, y_max]
	w_array = [x_min, x_max]

	h = np.diff(h_array)[0]
	w = np.diff(w_array)[0]

	cut = frame[y_min:y_min + h, x_min:x_min + w]
	
	cv2.imwrite(SAVE_PATH, cut)

# Capturing video frame
cap = cv2.VideoCapture(0)

while cap.isOpened():
    _ , frame = cap.read()
    # Cutting frame down to augmented size
    frame = frame[50:500, 50:500,:]
    
    # Change from BGR tot RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Resizing to 120 x 120
    resized = tf.image.resize(rgb, (120,120))

    # Scaling down and give it to the .predict() function of the facetracker
    yhat = facetracker.predict(np.expand_dims(resized/255,0))
    sample_coords = yhat[1][0]

    # Declaring waitkey
    k = cv2.waitKey(1)

    if yhat[0] > 0.5:
        # If pressed 's', and face is detected --> take cutted snapshot
        if k == ord('s'):
            saveImage(cap, sample_coords, frame)
            print('s')
            
        # Controls the main rectangle
        cv2.rectangle(frame, 
                      tuple(np.multiply(sample_coords[:2], [450,450]).astype(int)),
                      tuple(np.multiply(sample_coords[2:], [450,450]).astype(int)), 
                            (0,255,0), 2)

        # Controls the label rectangle
        cv2.rectangle(frame, 
                      tuple(np.add(np.multiply(sample_coords[:2], [450,450]).astype(int), 
                                    [0,-30])),
                      tuple(np.add(np.multiply(sample_coords[:2], [450,450]).astype(int),
                                    [80,0])), 
                            (0,255,0), -1)

        # Controls the text rendered
        cv2.putText(frame, 'face', tuple(np.add(np.multiply(sample_coords[:2], [450,450]).astype(int),
                                               [0,-5])),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

    cv2.imshow('FaceTrack', frame)

    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
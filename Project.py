#!/usr/bin/python
'''
	Author: Igor Maculan - n3wtron@gmail.com
	A Simple mjpg stream http server
'''
import cv2
from PIL import Image
import threading
from http.server import BaseHTTPRequestHandler,HTTPServer
from socketserver import ThreadingMixIn
from io import StringIO ,BytesIO
import time
capture=None

#!/usr/bin/env python
# coding: utf-8
"""
Object Detection (On Pi Camera) From TF2 Saved Model
=====================================
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
import pathlib
import tensorflow as tf
import cv2
import argparse
from threading import Thread

tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)
class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self ,framerate=30):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture("http://192.168.1.180:24861/videostream.cgi?user=admin&pwd=88888888")
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        #ret = self.stream.set(3,resolution[0])
        #ret = self.stream.set(4,resolution[1])
            
        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

	# Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
	# Start the thread that reads frames from the video stream
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
	# Return the most recent frame
        return self.frame
    
    def ret(self):
        return self.grabbed

    def stop(self):
	# Indicate that the camera and thread should be stopped
        self.stopped = True
        

parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Folder that the Saved Model is Located In',
                    default='ssd_mobilenet_v2_fpnlite_640x640')
parser.add_argument('--labels', help='Where the Labelmap is Located',
                    default='labelmap.pbtxt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.5)
                    
args = parser.parse_args()


# PROVIDE PATH TO MODEL DIRECTORY
PATH_TO_MODEL_DIR = args.model

# PROVIDE PATH TO LABEL MAP
PATH_TO_LABELS = args.labels

# PROVIDE THE MINIMUM CONFIDENCE THRESHOLD
MIN_CONF_THRESH = float(args.threshold)

# Load the model
# ~~~~~~~~~~~~~~
import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

PATH_TO_SAVED_MODEL = PATH_TO_MODEL_DIR + "/saved_model"

print('Loading model...')
start_time = time.time()

# Load saved model and build the detection function
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)

end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))

# Load label map data (for plotting)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                    use_display_name=True)

import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')   # Suppress Matplotlib warnings

print('Running inference for Camera')

import datetime
import requests

from ftplib import FTP
import os.path

ftp = FTP(host="192.168.1.57")
ftp.login('project','project')

class CamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.endswith('.mjpg'):
            self.send_response(200)
            self.send_header('Content-type','multipart/x-mixed-replace; boundary=--jpgboundary')
            self.end_headers()
            
            # used to record the time when we processed last frame 
            prev_frame_time = 0
  
            # used to record the time at which we processed current frame 
            new_frame_time = 0
            
            url = 'https://notify-api.line.me/api/notify'
            token = 'ACCagTTxrgFqU7Zkk8ZoU3UjQnqpIQLGDoe4eL3fOLn'
            headers = {'Authorization':'Bearer '+token}
            count_s = time.time()

            while True:
                try:
                    # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
                    # i.e. a single-column array, where each item in the column has the pixel RGB value
                    frame = videostream.read()
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    frame_expanded = np.expand_dims(frame_rgb, axis=0)
                    imH, imW, _ = frame.shape

                    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
                    input_tensor = tf.convert_to_tensor(frame)
                    # The model expects a batch of images, so add an axis with `tf.newaxis`.
                    input_tensor = input_tensor[tf.newaxis, ...]

                    # input_tensor = np.expand_dims(image_np, 0)
                    detections = detect_fn(input_tensor)

                    # All outputs are batches tensors.
                    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
                    # We're only interested in the first num_detections.
                    num_detections = int(detections.pop('num_detections'))
                    detections = {key: value[0, :num_detections].numpy()
                                   for key, value in detections.items()}
                    detections['num_detections'] = num_detections

                    # detection_classes should be ints.
                    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

                    
                    # SET MIN SCORE THRESH TO MINIMUM THRESHOLD FOR DETECTIONS
                    
                    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
                    scores = detections['detection_scores']
                    boxes = detections['detection_boxes']
                    classes = detections['detection_classes']
                    count = 0
                    
                    count_c = time.time()
                    for i in range(len(scores)):
                        if ((scores[i] > MIN_CONF_THRESH) and (scores[i] <= 1.0)):
                            #increase count
                            count += 1
                            # Get bounding box coordinates and draw box
                            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                            ymin = int(max(1,(boxes[i][0] * imH)))
                            xmin = int(max(1,(boxes[i][1] * imW)))
                            ymax = int(min(imH,(boxes[i][2] * imH)))
                            xmax = int(min(imW,(boxes[i][3] * imW)))
                            
                            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
                            # Draw label
                            object_name = category_index[int(classes[i])]['name'] # Look up object name from "labels" array using class index
                            label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1) # Get font size
                            label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                            cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1) # Draw label text
                            
                            if count_c - count_s > 10: #sec
                                count_s = time.time()
                                time_stamp = datetime.datetime.fromtimestamp(count_s).strftime('%Y-%m-%d %H:%M:%S')
                                cv2.imwrite("detected_img/tmp.jpg",frame)
                                
                                fp = open("detected_img/tmp.jpg", "rb")
                                ftp.storbinary("STOR %s" % os.path.basename(time_stamp + '.jpg'), fp)
                                fp.close()
                                
                                requests.post(url, headers=headers ,data = {'message':'Detected '+str(object_name)+' at : '+ time_stamp} ,files = {'imageFile':open("detected_img/tmp.jpg",'rb')})

                    new_frame_time = time.time() 
                  
                    # Calculating the fps 
                  
                    # fps will be number of frame processed in given time frame 
                    # since their will be most of time error of 0.001 second 
                    # we will be subtracting it to get more accurate result 
                    fps = 1/(new_frame_time-prev_frame_time) 
                    prev_frame_time = new_frame_time 
                  
                    # converting the fps into integer 
                    fps = int(fps) 
                  
                    # converting the fps to string so that we can display it on frame 
                    # by using putText function 
                    fps = str(fps) 
                  
                    # puting the FPS count on the frame
                    cv2.putText(frame, 'FPS : ' + fps, (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (70,235,52), 1, cv2.LINE_AA)

                    cv2.putText (frame,'Objects Detected : ' + str(count),(5,15),cv2.FONT_HERSHEY_SIMPLEX, 0.4,(70,235,52),1,cv2.LINE_AA)

                    imgRGB=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                    jpg = Image.fromarray(imgRGB)
                    tmpFile = BytesIO()
                    jpg.save(tmpFile,'JPEG')
                    self.wfile.write("--jpgboundary".encode())
                    self.send_header('Content-type','image/jpeg')
                    self.send_header('Content-length',str(tmpFile.getbuffer().nbytes))
                    self.end_headers()
                    jpg.save(self.wfile,'JPEG')
                    time.sleep(0.05)
                except KeyboardInterrupt:
                    break
            return
        if self.path.endswith('.html'):
            self.send_response(200)
            self.send_header('Content-type','text/html')
            self.end_headers()
            self.wfile.write('<html><head></head><body>'.encode())
            self.wfile.write('<img src="http://127.0.0.1:8080/cam.mjpg"/>'.encode())
            self.wfile.write('</body></html>'.encode())
            return


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread."""

def main():
    global videostream
    videostream = VideoStream(framerate=30).start()
    try:
        server = ThreadedHTTPServer(('192.168.1.115', 8080), CamHandler)
        print('server started')
        server.serve_forever()
    except KeyboardInterrupt:
        capture.release()
        server.socket.close()

if __name__ == '__main__':
    main()



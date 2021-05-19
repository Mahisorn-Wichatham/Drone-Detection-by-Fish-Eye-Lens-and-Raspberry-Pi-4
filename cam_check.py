import cv2
import numpy as np
import time

# your video stream path
cap = cv2.VideoCapture("http://192.168.1.180:24861/videostream.cgi?user=admin&pwd=88888888")

# used to record the time when we processed last frame 
prev_frame_time = 0
  
# used to record the time at which we processed current frame 
new_frame_time = 0

while(True):

    ret, frame = cap.read()
    
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
    cv2.putText(frame, 'FPS : ' + fps, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 1, cv2.LINE_AA)
    
    cv2.putText (frame,'Objects Detected : ' + str(2),(10,25),cv2.FONT_HERSHEY_SIMPLEX,1,(70,235,52),1,cv2.LINE_AA)
    
    cv2.imshow('frame',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):

        break

cap.release()

cv2.destroyAllWindows()
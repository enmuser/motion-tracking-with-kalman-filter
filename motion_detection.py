# this algorithm allows us to detect a motion in a video from camera
# and traking object
# importing of package

from imutils.video import VideoStream
from collections import deque
import matplotlib.pyplot as plt
from pykalman import KalmanFilter
import numpy as np
import imutils
import datetime
import argparse
import time
import cv2 

# consturct the argement parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="Path to the video file")
ap.add_argument("-b", "--buffer", type=int, default=0, help="max buffer size")
ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")
args = vars(ap.parse_args())

#####################################################################

# return centre of a set of points representing a rectangle

def center(points):
    x = np.float32((points[0][0] + points[1][0] + points[2][0] + points[3][0]) / 4.0)
    y = np.float32((points[0][1] + points[1][1] + points[2][1] + points[3][1]) / 4.0)
    return np.array([np.float32(x), np.float32(y)], np.float32)

#####################################################################

# init kalman filter object

kalman = cv2.KalmanFilter(4,2)
kalman.measurementMatrix = np.array([[1,0,0,0],
                                     [0,1,0,0]],np.float32)

kalman.transitionMatrix = np.array([[1,0,1,0],
                                    [0,1,0,1],
                                    [0,0,1,0],
                                    [0,0,0,1]],np.float32)

kalman.processNoiseCov = np.array([[1,0,0,0],
                                   [0,1,0,0],
                                   [0,0,1,0],
                                   [0,0,0,1]],np.float32) * 0.03

measurement = np.array((2,1), np.float32)
prediction = np.zeros((2,1), np.float32)

print("\nObservation in image: BLUE")
print("Prediction from Kalman: GREEN")
print("Update from Kalman: RED\n")

# if the video argument is empty, then we are reading from camera
if args.get("video", None) is None:
    vs = VideoStream(src=0).start()
    with_plt = 480
    height_plt = 360
    time.sleep(2.0)
else:
    vs = cv2.VideoCapture(args["video"])
    print("\t Width: ",vs.get(cv2.CAP_PROP_FRAME_WIDTH))
    print("\t Height: ",vs.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("\t FourCC: ",vs.get(cv2.CAP_PROP_FOURCC))
    print("\t Framerate: ",vs.get(cv2.CAP_PROP_FPS))

    with_plt = vs.get(cv2.CAP_PROP_FRAME_WIDTH)
    height_plt = vs.get(cv2.CAP_PROP_FRAME_HEIGHT)
# initiaze the first frame in the video stream
firstFrame = None

plt.figure()
#plt.hold(True)

plt.axis([0, with_plt, height_plt, 0])
# Setup the termination criteria, either 10 iteration or move by atleast 1 pt

term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
numframes = int(vs.get(7))
measuredTrack = np.zeros((numframes,2))-1
count = 0 
id  = []
# loop over the frames of the video

while True:
    # grab the current frame and initialize the occupied / unoccupied text
    count += 1
    frame = vs.read()
    frame = frame if args.get("video", None) is None else frame[1]
    
    # if the frame could not be grabed, then we have reached the end of the video
    if frame is None:
        break
    
    # resize the frame, convert it to grayscale, and blur it
    frame = imutils.resize(frame, width = int(with_plt))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
     
    # if the first frame is None, initialize it
    if firstFrame is None:
        firstFrame = gray
        continue
    
    # comptute the aboslute difference between the current frame and first frame
    frameDelta = cv2.absdiff(firstFrame, gray)
    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
    
    # dilate and erosions to remove any small blobs who lefts
    # then find contours on threshlolded image
    
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
        
    # loop over the countours
    
    for c in range(len(cnts)):
        # if the contour is too small, ignor it
        if cv2.contourArea(cnts[c]) < args["min_area"]:
            continue
        
        # compute the bounding box for the  contour, draw it on the frame
        track_win = cv2.boundingRect(cnts[c])
        
        #ret, track_win = cv2.CamShift(thresh, track_win, term_crit)
        
        (x, y, w, h) = track_win
        cv2.rectangle(frame, (x,y), (x + w, y + h), (0, 255, 0 ), 2)
        cv2.putText(frame, "Id : {}".format(c + 1) , (x , y  - 3), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 2)
        
        # extract center of this obsevation as point
        rect = cv2.minAreaRect(cnts[c])
        pts = cv2.boxPoints(rect)
        pts = np.int0(pts)
        measuredTrack[count-1,:] = pts[0]
        plt.plot(pts[0,0],pts[0,1], 'xg')
        
        # use to correct the Kalman filter
        #kalman.correct(center(pts))
        
        # get new Kalman filter prediction
        #prediction = kalman.predict()
        
        # draw prediction on image in BLUE
        #cv2.rectangle(frame, (prediction[0]-(0.5*w),prediction[1]-(0.5*h)), (prediction[0]+(0.5*w),prediction[1]+(0.5*h)), (255,0,0),2)

        #plt.plot(prediction[0],prediction[1], 'xb')
        # get new Kalman filter update
        
       # update = kalman.update()
        
    # draw the text and timestamp on the frame
    cv2.putText(frame, "Object found: {}".format(len(cnts)), (10, 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),2)
    cv2.putText(frame, datetime.datetime.now().strftime("%A %B %Y %I:%M:%S%p"),
                (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
    # show the frame and record if the user presses a key
    cv2.imshow("Main Frame", frame)
    cv2.waitKey(200)
    cv2.imshow("Main Frame", frame)
    cv2.moveWindow("Main Frame", 0,0)
    cv2.imshow("Seuil", thresh)
    cv2.moveWindow("Seuil", 600,0)
    
   # cv2.imshow("Frame Delta", frameDelta)
   # cv2.moveWindow("Frame Delta",700,0)
    key = cv2.waitKey(1) & 0xFF
    # if the `q` key is pressed, break from the lop
    if key == ord("q"):
        break
# cleanup the camera and close any open windows
plt.savefig('kalmanfilter.png')
np.save("motionTrajector", measuredTrack)
plt.show()
vs.stop() if args.get("video", None) is None else vs.release()
cv2.destroyAllWindows()
    
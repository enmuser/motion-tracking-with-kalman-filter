# Motion Tracking With Python, Opencv and Kalman Filter

In this projet, i try to tracking motion with opencv 3.4
For that i use background subtraction.

## Background subtraction
Background subtraction is critical in many computer vision applications.

We use it to count the number of cars passing through a toll booth. 

We use it to count the number of people walking in and out of a store.

And we use it for motion detection. We use the first image of video and we maintain it static

with black color, and for the erea where the emotion is detected we can see white color

![Background](https://github.com/Stevencibambo/motion-tracking-with-kalman-filter/blob/master/images/track2.png)

![Background](https://github.com/Stevencibambo/motion-tracking-with-kalman-filter/blob/master/images/trac11.png)

![Background](https://github.com/Stevencibambo/motion-tracking-with-kalman-filter/blob/master/images/track12.png)

## For Tracking we use Kalman Filter

With Kalman Filter we predict and update the meared position.

![Measured position](https://github.com/Stevencibambo/motion-tracking-with-kalman-filter/blob/master/images/kalmanfilter1.png)
![Output Kalman](https://github.com/Stevencibambo/motion-tracking-with-kalman-filter/blob/master/images/outputkalmanfilter.png)

## Runing
With Python 3
python motion_detection.py --video video/path_video

or 
to use webcom or camera

python motion_detection.py

The measured positions are saved in motionTrajector.npy
To predict and update
run
python measured.py

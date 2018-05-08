# Motion Detection
#### Motivation
The target of our project is to implement a traffic condition estimating system which is able to detect the movements of vehicles in road, track cars and to estimate the road conditions by analyzing videos/stream from monitoring cameras.
#### Design
Our system consists of three parts: Video/Stream Processing，Vehicles Detection and Road Condition Estimation. Object detection plays a key role in image recognition system and can be applied in different application, ranging from monitoring to advance tracking. In our project, we are going to use optical flow to detect moving object. We then estimate the traffic condition based on detected data like the number of vehicles in screen at the same time, average moving speed of vehicles comparing to the speed limits of the road, etc.

#### How to run the program?
```
python test.py
```

* You could to change the params for scene function in order to run HS or LK.
* The default speed limit for the road is 40 miles/hour, once the vehicle speed exceeds the limit, the rectangle will turn to red.
* See result in ./res/output.avi




# Hand Gesture Controlled Volume

This Python script uses computer vision and hand tracking to control the system's volume based on hand gestures captured through the webcam.

## Pre Requisites

- Python 3.x
- OpenCV (`cv2`)
- Mediapipe (`mediapipe`)
- Pycaw (`pycaw`)
- Numpy (`numpy`)

Install the required libraries using the following command:
```bash
pip install opencv-python mediapipe pycaw numpy
```

## Usage

Run the script (hand_volume_control.py).
Ensure your camera is enabled and directed towards your hand.
Perform hand gestures:
Extend your thumb and index finger.
Adjust the distance between thumb and index finger to control the volume level.
The program will print the calculated volume level based on hand distance.
Press the 'q' key to quit the program.

## Code Explanation
The script performs the following actions:

Initializes the camera for capturing video.
Utilizes the mediapipe library to detect and track hand landmarks.
Maps the distance between thumb and index finger to adjust system volume.
Displays a volume bar visualization on the video feed.

### Dependencies
The script relies on the following libraries:
<pre>
Cv2       : OpenCV for capturing and processing video frames.
Mediapipe : For hand tracking and landmark detection.
Pycaw     : To access system volume controls.
Numpy     : Used for data manipulation and interpolation.
</pre>

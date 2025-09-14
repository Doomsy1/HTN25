we have a projector and a laptop and a raspberry pi 5, the raspberry pi 5 has 2 picams attached.

laptop: windows 11, python 3.12.11

Turn any projector into a touch surface using **two Pi cameras** and a **Windows laptop**. Detect a fingertip in stereo, intersect it with the screen plane, and inject **native Windows touch** events (not mouse). - fallback to mouse if we have problems with touch

to calibrate, the projector will display a png of mostly white screen with aruco markers to be seen by the picams

the picams will each take an image and use the positions of the markers to calculate the stereo calibration as well as the screen plane and the projector to screen mapping

using the calibration, the projected screen (aruco markers) should be subtracted from the laptop screen to get everything that isn't screen - which would be a human in front of the projected screen or anything else that is not part of the screen.
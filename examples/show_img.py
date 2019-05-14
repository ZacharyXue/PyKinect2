from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime

import numpy as np

import cv2

class kinect_rgb(object):
    def __init__(self):
        
        # Kinect runtime object, we want only color and body frames 
        # use this to create the connection with Kinect
        self._kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Body)

    def run(self):
        # -------- Main Program Loop -----------
        while cv2.waitKey(10) != 'q':

            # --- Getting frames and drawing   
            if self._kinect.has_new_color_frame():
                frame = self._kinect.get_last_color_frame()
                # print("The shape of frame is {}".format(frame.shape))
                frame = np.reshape(frame,[1080,1920,4])
                # frame2 = np.array(frame,dtype=np.uint8)
                img = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
                cv2.imshow('rgb',img)
                frame = None


        # Close our Kinect sensor, close the window and quit.
        self._kinect.close()


__main__ = "show_img"
game = kinect_rgb();
game.run();


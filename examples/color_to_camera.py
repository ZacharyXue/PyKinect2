from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime

import numpy as np

import cv2

import ctypes
import time

# https://github.com/starsoi/vecathon2018/blob/2d21f5866f0357a425dce5306131222bfb9fd390/kinect/main.py
# https://github.com/aliceyew/SmartCity/blob/76f98f88955061a74ce0bee063350b31d902d26e/Kinect%20code.py
# https://bbs.csdn.net/topics/392068261?page=1

kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Depth)

width, height = kinect.color_frame_desc.width, kinect.color_frame_desc.height
cameraPoint_capacity = ctypes.c_uint(width * height)
cameraPoint_data_type = PyKinectV2._CameraSpacePoint * cameraPoint_capacity.value

while 1:    
    if kinect.has_new_depth_frame():   
        cameraPointCount = ctypes.POINTER(PyKinectV2._CameraSpacePoint)
        
        cameraPointCount = ctypes.cast(cameraPoint_data_type(), ctypes.POINTER(PyKinectV2._CameraSpacePoint))   
        hr = kinect._mapper.MapColorFrameToCameraSpace(kinect._depth_frame_data_capacity,kinect._depth_frame_data,\
            cameraPoint_capacity,cameraPointCount)
        print(float(cameraPointCount[int(0.5*height*width + 0.5*width)].x))
        # I guess the result is inf becuase the Kinect could not detect in that distance
kinect.close()
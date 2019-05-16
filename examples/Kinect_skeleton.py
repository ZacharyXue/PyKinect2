from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime

import numpy as np

import cv2

class kinect_rgbd(object):
    def __init__(self):
        
        # Kinect runtime object, we want only color and body frames 
        # use this to create the connection with Kinect
        self._kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Body)
       # The function draw_body_bone and draw_body should just copy to other codes
    def draw_body_bone(self, joints, jointPoints, joint0, joint1):
        joint0State = joints[joint0].TrackingState;
        joint1State = joints[joint1].TrackingState;

        # both joints are not tracked
        if (joint0State == PyKinectV2.TrackingState_NotTracked) or (joint1State == PyKinectV2.TrackingState_NotTracked): 
            return

        # both joints are not *really* tracked
        if (joint0State == PyKinectV2.TrackingState_Inferred) and (joint1State == PyKinectV2.TrackingState_Inferred):
            return

        start = (int(jointPoints[joint0].x), int(jointPoints[joint0].y))
        end = (int(jointPoints[joint1].x), int(jointPoints[joint1].y))

        cv2.line(self.frame,start,end,(0, 0, 255),8)

    def draw_body(self, joints, jointPoints):
        # Torso
        self.draw_body_bone(joints, jointPoints ,PyKinectV2.JointType_Head, PyKinectV2.JointType_Neck)
        self.draw_body_bone(joints, jointPoints, PyKinectV2.JointType_Neck, PyKinectV2.JointType_SpineShoulder)
        self.draw_body_bone(joints, jointPoints,  PyKinectV2.JointType_SpineShoulder, PyKinectV2.JointType_SpineMid)
        self.draw_body_bone(joints, jointPoints,  PyKinectV2.JointType_SpineMid, PyKinectV2.JointType_SpineBase)
        self.draw_body_bone(joints, jointPoints, PyKinectV2.JointType_SpineShoulder, PyKinectV2.JointType_ShoulderRight)
        self.draw_body_bone(joints, jointPoints,  PyKinectV2.JointType_SpineShoulder, PyKinectV2.JointType_ShoulderLeft)
        self.draw_body_bone(joints, jointPoints,  PyKinectV2.JointType_SpineBase, PyKinectV2.JointType_HipRight)
        self.draw_body_bone(joints, jointPoints, PyKinectV2.JointType_SpineBase, PyKinectV2.JointType_HipLeft)
    
        # Right Arm    
        self.draw_body_bone(joints, jointPoints, PyKinectV2.JointType_ShoulderRight, PyKinectV2.JointType_ElbowRight)
        self.draw_body_bone(joints, jointPoints,  PyKinectV2.JointType_ElbowRight, PyKinectV2.JointType_WristRight)
        self.draw_body_bone(joints, jointPoints, PyKinectV2.JointType_WristRight, PyKinectV2.JointType_HandRight)
        self.draw_body_bone(joints, jointPoints,PyKinectV2.JointType_HandRight, PyKinectV2.JointType_HandTipRight)
        self.draw_body_bone(joints, jointPoints, PyKinectV2.JointType_WristRight, PyKinectV2.JointType_ThumbRight)

        # Left Arm
        self.draw_body_bone(joints, jointPoints,  PyKinectV2.JointType_ShoulderLeft, PyKinectV2.JointType_ElbowLeft)
        self.draw_body_bone(joints, jointPoints,  PyKinectV2.JointType_ElbowLeft, PyKinectV2.JointType_WristLeft)
        self.draw_body_bone(joints, jointPoints,  PyKinectV2.JointType_WristLeft, PyKinectV2.JointType_HandLeft)
        self.draw_body_bone(joints, jointPoints,  PyKinectV2.JointType_HandLeft, PyKinectV2.JointType_HandTipLeft)
        self.draw_body_bone(joints, jointPoints,  PyKinectV2.JointType_WristLeft, PyKinectV2.JointType_ThumbLeft)

        # Right Leg
        self.draw_body_bone(joints, jointPoints,  PyKinectV2.JointType_HipRight, PyKinectV2.JointType_KneeRight)
        self.draw_body_bone(joints, jointPoints,  PyKinectV2.JointType_KneeRight, PyKinectV2.JointType_AnkleRight)
        self.draw_body_bone(joints, jointPoints,  PyKinectV2.JointType_AnkleRight, PyKinectV2.JointType_FootRight)

        # Left Leg
        self.draw_body_bone(joints, jointPoints,  PyKinectV2.JointType_HipLeft, PyKinectV2.JointType_KneeLeft)
        self.draw_body_bone(joints, jointPoints,  PyKinectV2.JointType_KneeLeft, PyKinectV2.JointType_AnkleLeft)
        self.draw_body_bone(joints, jointPoints,  PyKinectV2.JointType_AnkleLeft, PyKinectV2.JointType_FootLeft)



    def run(self):
        # -------- Main Program Loop -----------
        while cv2.waitKey(10) != 27:

            # --- Getting frames and drawing   
            if self._kinect.has_new_color_frame():
                frame = self._kinect.get_last_color_frame()
                # print("The shape of frame is {}".format(frame.shape))
                frame = np.reshape(frame,[1080,1920,4])
                self.frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

                # --- Cool! We have a body frame, so can get skeletons 
                if self._kinect.has_new_body_frame(): 
                    self._bodies = self._kinect.get_last_body_frame()

                    # --- draw skeletons to _frame_surface
                    if self._bodies is not None: 
                        for i in range(0, self._kinect.max_body_count):
                            body = self._bodies.bodies[i]
                            if not body.is_tracked: 
                                continue 
                            
                            joints = body.joints 
                            # convert joint coordinates to color space 
                            joint_points = self._kinect.body_joints_to_color_space(joints)
                            self.draw_body(joints, joint_points)

                cv2.imshow('rgbd',self.frame)

        # Close our Kinect sensor, close the window and quit.
        self._kinect.close()


test = kinect_rgbd()
test.run()
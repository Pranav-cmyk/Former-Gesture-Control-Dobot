from .handtracking import HandTracking
from dotenv import load_dotenv
from typing import Tuple
from pydobot import Dobot

import cv2 as cv
import os
import logging
import numpy as np
import time

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class robotMovementError(Exception):
    pass


class RobotController:
    def __init__(self, port: str = None, cameraIndex: int = 0, numberOfHands: int = 1):

        if not port:
            port = os.environ.get("DOBOT_PORT")
        
        print(f"Connecting to Dobot on port: {port}...")
        self.camera = cv.VideoCapture(cameraIndex)
        self.dobot = Dobot(port, verbose=True)
        self.tracker = HandTracking(max_num_hands=numberOfHands)

    def getJointPositions(self):
        return self.dobot.pose()[4:8]

    def updateRobotPosition(self, BGRFrame: np.ndarray, *jointAngles: float):

        RGBFrame = cv.cvtColor(BGRFrame, cv.COLOR_BGR2RGB)
        BGRFrame, handlms = self.tracker.detectHands(RGBFrame)
        if len(handlms) > 0:
            BGRFrame, angles = self.tracker.getAnglesFromLandmarks(BGRFrame, handlms, 8, 5, 0)
        
        # move only the targeted Joint
            try:
                self.dobot._set_ptp_cmd(angles[0], jointAngles[1], jointAngles[2], jointAngles[3],
                    mode=4,
                    wait=True
                )
            except Exception as e:
                raise robotMovementError("Failed to update robot position: {}".format(e))

        time.sleep(0.05)
        return BGRFrame

    def start(self):
        print("Starting robot controller...")
        if not self.camera.isOpened():
            logger.error("Could not open camera.")
            return

        try:
            j1, j2, j3, j4 = self.getJointPositions()
        except Exception as e:
            logger.error(f"Could not get initial joint positions from Dobot: {e}")
            return

        while self.camera.isOpened():
            try:
                success, frame = self.camera.read()
                if not success:
                    logger.warning("Failed to read frame from camera.")
                    break

                BGRFrame = self.updateRobotPosition(frame, j1, j2, j3, j4)     
                cv.imshow("Frame", cv.flip(BGRFrame, 1))
            
                if cv.waitKey(1) & 0xFF == ord('q'):
                    logger.info("Exiting main loop")
                    break
                    
            except Exception as e:
                logger.error("An error occurred: {}".format(e))
                raise e

        cv.destroyAllWindows()
        self.camera.release()
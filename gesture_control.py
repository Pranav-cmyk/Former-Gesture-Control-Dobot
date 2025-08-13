from src.handtracking import HandTracking
from pydobot import Dobot
import cv2 as cv


cap = cv.VideoCapture(0)
tracker = HandTracking(max_num_hands=2)
dobot = Dobot(port='COM5') # Adjust Port from device manager

while True:
    try:
        _, frame = cap.read()
        RGBFrame = cv.cvtColor(frame, cv.COLOR_BGR2RGB) 
        BGRFrame, handlms = tracker.detecthands(RGBFrame)
            
        if len(handlms) > 1:
            _, distance = tracker.Drawconnections(BGRFrame, handlms[0], 4, 20)
            _, left_angles = tracker.get_angles_from_lndmks(BGRFrame, handlms, 8, 5, 0)
            _, right_angles = tracker.get_angles_from_lndmks(BGRFrame, handlms, 12, 9, 0)
            dobot._set_ptp_cmd(left_angles[0], left_angles[1], right_angles[1], 0, mode=4, wait=True)
    

            if distance < 30:
                dobot.suck(True)
            else:
                dobot.suck(False)


        BGRFrame = cv.flip(BGRFrame, 1)
        cv.imshow("Frame", BGRFrame)
        
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    except Exception as e:
        print(e)
        
cv.destroyAllWindows()
dobot._set_ptp_cmd(0, 0, 0, 0, mode=4, wait=True)
dobot.close()
cap.release()

            
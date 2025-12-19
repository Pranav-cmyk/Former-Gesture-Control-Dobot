from src.handtracking import HandTracking
import cv2 as cv


cap = cv.VideoCapture(0)
tracker = HandTracking(max_num_hands=2)

while True:
    try:
        _, frame = cap.read()
        RGBFrame = cv.cvtColor(frame, cv.COLOR_BGR2RGB) 
        BGRFrame, handlms = tracker.detecthands(RGBFrame)
        BGRFrame = cv.flip(BGRFrame, 1)
        cv.imshow("Frame", BGRFrame)
        
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    except Exception as e:
        print(e)
        
cv.destroyAllWindows()
cap.release()

            
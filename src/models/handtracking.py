import mediapipe as mp
import cv2 as cv
from math import sqrt, acos, degrees


class HandTracking:
    def __init__(self, min_detection_confidence = 0.5, min_tracking_confidence = 0.5, max_num_hands = 1):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands_instance = self.mp_hands.Hands(min_detection_confidence = min_detection_confidence, min_tracking_confidence = min_tracking_confidence, max_num_hands = max_num_hands)
        
    def detectHands(self, RGBframe):
        result = self.hands_instance.process(RGBframe)
        handslms = []
        if result.multi_hand_landmarks:
            for handlmk in result.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(RGBframe, handlmk, self.mp_hands.HAND_CONNECTIONS)
                handslms.append(handlmk)
        return cv.cvtColor(RGBframe, cv.COLOR_RGB2BGR), handslms
    
    def getCoordinates(self, hand, *marks, frame):
        height, width = frame.shape[:2]
        pcoords = {}
        for mark in marks:
            point = hand.landmark[mark]
            pcoords[mark] = (int(point.x * width), int(point.y * height))
        return pcoords
        
    def drawConnections(self, frame, hand, mark1, mark2):
        h, w = frame.shape[:2]
        distance = None
        ppoints = self.getCoordinates(hand, mark1, mark2, frame = frame)
        (x1, y1), (x2, y2) = ppoints[mark1], ppoints[mark2]
        cv.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)
        distance = sqrt((x1 - x2)**2 + (y1 - y2)**2)
        return frame, distance
    
    def getAnglesFromLandmarks(self, frame, hands, mark1=8, mark2=5, mark3=0):
        angles = []
        for hand in hands:
            try:
                points = self.getCoordinates(hand, mark1, mark2, mark3, frame = frame)
                (x1, y1), (x2, y2), (x3, y3) = points[mark1], points[mark2], points[mark3]
                a = sqrt((x1 - x2)**2 + (y2 - y1)**2)
                b = sqrt((x3 - x2)**2 + (y3 - y2)**2)
                c = sqrt((x1 - x3)**2 + (y3 - y1)**2)
                angle = degrees(acos((a**2 + b**2 - c**2) / (2 * a * b)))
                angles.append(180 - angle)
            except Exception as e:
                angles.append(0)
                
        return frame, angles
        
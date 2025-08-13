import mediapipe as mp
import cv2 as cv
from math import sqrt, acos, degrees


class HandTracking:
    def __init__(self, min_detection_confidence = 0.5, min_tracking_confidence = 0.5, max_num_hands = 1):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands_instance = self.mp_hands.Hands(min_detection_confidence = min_detection_confidence, min_tracking_confidence = min_tracking_confidence, max_num_hands = max_num_hands)
        
    def detecthands(self, RGBframe):
        result = self.hands_instance.process(RGBframe)
        handslms = []
        if result.multi_hand_landmarks:
            for handlmk in result.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(RGBframe, handlmk, self.mp_hands.HAND_CONNECTIONS)
                handslms.append(handlmk)
        return cv.cvtColor(RGBframe, cv.COLOR_RGB2BGR), handslms
    
    def get_coordinates(self, hand, *marks, frame):
        height, width = frame.shape[:2]
        ncoords = {}
        pcoords = {}
        for mark in marks:
            point = hand.landmark[mark]
            ncoords[mark] = (point.x, point.y)
            pcoords[mark] = (int(point.x * width), int(point.y * height))
        return ncoords, pcoords
        
    def Drawconnections(self, frame, hand, mark1, mark2, use_normalised_distance = False):
        h, w = frame.shape[:2]
        distance = None
        
        npoints, ppoints = self.get_coordinates(hand, mark1, mark2, frame = frame)
        (x1, y1), (x2, y2) = ppoints[mark1], ppoints[mark2]
        cv.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)
        if use_normalised_distance:
            (x1, y1), (x2, y2) = npoints[mark1], npoints[mark2]
        distance = sqrt((x1 - x2)**2 + (y1 - y2)**2)

        return frame, distance
    
    def get_angles_from_lndmks(self, frame, hands, mark1, mark2, mark3):
        if len(hands) == 0:
            return frame, []
        angles = []
        
        for hand in hands:
            try:
                _, ppoints = self.get_coordinates(hand, mark1, mark2, mark3, frame = frame)
                (x1, y1), (x2, y2), (x3, y3) = ppoints[mark1], ppoints[mark2], ppoints[mark3]
                a = sqrt((x1 - x2)**2 + (y2 - y1)**2)
                b = sqrt((x3 - x2)**2 + (y3 - y2)**2)
                c = sqrt((x1 - x3)**2 + (y3 - y1)**2)
                angle = degrees(acos((a**2 + b**2 - c**2) / (2 * a * b)))
                angles.append(180 - angle)
            except Exception as e:
                print(e)
                angles.append(0)
                
        return frame, angles
        
    def Count_Fingers(self, frame, hands, text_location = (), images = None, use_overlay_images = False):
        
        if len(hands) == 0:
            return frame, 0

        finger_count = 0
        finger_tips = [8, 12, 16, 20]
        finger_bases = [6, 10, 14, 18]

        for hand in hands:
            _, pcoords = self.get_coordinates(hand, *finger_tips, *finger_bases, frame=frame)
            
            for tip, base in zip(finger_tips, finger_bases):
                if pcoords[tip][1] < pcoords[base][1]:
                    finger_count += 1

        if use_overlay_images:
            if images is not None:
                overlay_image = images[min(finger_count, len(images) - 1)]
                h, w = overlay_image.shape[:2]
                frame[0:h, 0:w] = overlay_image
            else:
                raise Exception("Overlay images not provided")

        cv.putText(frame, f"I see {finger_count} fingers", text_location, cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4)
        
        return frame, finger_count


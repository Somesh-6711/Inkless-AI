import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

class HandTracker:
    def __init__(self, max_hands=1, detection_conf=0.7, tracking_conf=0.7):
        """
        Initializes the hand tracking model.
        :param max_hands: Maximum number of hands to detect
        :param detection_conf: Minimum confidence for detection
        :param tracking_conf: Minimum confidence for tracking
        """
        self.hands = mp_hands.Hands(
            max_num_hands=max_hands,
            min_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf
        )

    def track_hands(self, frame):
        """
        Processes the frame to detect hands and returns annotated frame.
        :param frame: Input frame from webcam
        :return: Processed frame with hand landmarks
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        return frame
